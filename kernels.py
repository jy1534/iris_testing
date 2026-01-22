from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from .utils import ShmemBuffers, build_expert_offsets

import iris

"""
world = DeviceCount E = E_local = T 
H = hidden_dimension 
CAP = max_{e,s} pca[e,s] 
Capcity in real moe 
"""

# Step-1 kernel: exchange token counts. 
@triton.jit
def counts_exchange_kernel(
    send_counts_ptr,  # [world, E] int32 (local)
    pca_ptr,  # [E, world] int32 (symmetric on dst)
    counts_ready_ptr,  # [1] int32 (symmetric on dst)
    heap_bases,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Write counts to each dst's PCA[:, src_rank], then signal counts_ready++ on dst."""

    dst = tl.program_id(0)  # one program per destination rank

    # Write the E counts for this destination.
    for e0 in tl.static_range(0, e_local, BLOCK_E):
        e = e0 + tl.arange(0, BLOCK_E)
        mask_e = e < e_local

        # Local read: send_counts[dst, e]
        vals = tl.load(send_counts_ptr + dst * e_local + e, mask=mask_e, other=0).to(tl.int32)

        # Remote write: pca[e, src_rank] on destination.
        remote_ptr = pca_ptr + e * world_size + src_rank
        iris.put(
            remote_ptr,
            vals,
            src_rank,
            dst,
            heap_bases,
            mask=mask_e,
        )

    # Signal completion to destination (release semantics).
    iris.atomic_add(
        counts_ready_ptr,
        1,
        src_rank,
        dst,
        heap_bases,
        sem="release",
        scope="sys",
    )

# Step-2 kernel: token exchange (with local spin-wait on counts_ready).
@triton.jit
def tokens_exchange_kernel(
    counts_ready_ptr,  # [1] int32 local
    send_ptr,
    send_counts_ptr,
    dst_offsets_ptr,
    expert_offs_ptr,
    token_buf_ptr,
    token_sync_ptr,
    heap_bases,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
    CAP: tl.constexpr,
    hidden_dim: tl.constexpr,
    EXPECTED: tl.constexpr,   # NEW
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dst = tl.program_id(0) # destination rank
    expert = tl.program_id(1) # destination-local expert index

    # Only enforce local acquire on the receiver-side programs (dst == this rank)
    if dst == src_rank:
        while tl.load(counts_ready_ptr, volatile=True).to(tl.int32) < EXPECTED:
            pass

        # Acquire fence (local): pair with remote release increments on dst counts_ready.
        iris.atomic_add(
            counts_ready_ptr,
            0,
            src_rank,
            src_rank,
            heap_bases,
            sem="acquire",
            scope="gpu",
        )
    # How many rows to send for this (dst, expert)
    n = tl.load(send_counts_ptr + dst * e_local + expert).to(tl.int32)

    # Base row offset into packed send_ptr for this (dst, expert)
    dst_base = tl.load(dst_offsets_ptr + dst).to(tl.int32)
    e_off = tl.load(expert_offs_ptr + dst * e_local + expert).to(tl.int32)
    send_base = dst_base + e_off

    # Remote base for token_buf[expert, src_rank, 0, 0] on destination
    # Flatten: (((expert * world + src_rank) * CAP + m) * H + k)
    remote_base = (expert * world_size + src_rank) * CAP * hidden_dim

    # Copy up to CAP rows; rows beyond n are masked (padding remains zero)
    for m0 in tl.static_range(0, CAP, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        m_mask = offs_m < n
        row_ids = (send_base + offs_m).to(tl.int32)

        for k0 in tl.static_range(0, tl.cdiv(hidden_dim, BLOCK_K)):
            offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < hidden_dim

            send_ptrs = send_ptr + row_ids[:, None] * hidden_dim + offs_k[None, :]
            x = tl.load(send_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            remote_ptrs = token_buf_ptr + remote_base + offs_m[:, None] * hidden_dim + offs_k[None, :]
            iris.put(
                remote_ptrs,
                x,
                src_rank,
                dst,
                heap_bases,
                mask=m_mask[:, None] & k_mask[None, :],
            )

    # Signal completion for this expert on destination (release semantics).
    # IMPORTANT: even if n == 0, we still increment once so dst can reach world_size.
    iris.atomic_add(
        token_sync_ptr + expert,
        1,
        src_rank,
        dst,
        heap_bases,
        sem="release",
        scope="sys",
    )

@triton.jit
def token_shuffle(
    pca_cumsum_ptr, pca_ptr, ## Both of size: [E, world_size]
    token_buffer_ptr, # Size: [E, world_size, mxa, hidden_dim]
    output_buffer_ptr, # Size: [S, hidden_dim]
    token_sync_ptr, # Size: [E] int32 
    E: tl.constexpr, world_size: tl.constexpr,
    mxa: tl.constexpr, hidden_dim: tl.constexpr
    BLOCK_X: tl.contexpr  ## We have 1-d blocks only.
):
    """
    Triton kernel that reshuffles data post all-to-all (prior to expert compute) 
    to eliminate zero-padding.

    Args:
        pca_cumsum_ptr (Tensor): [E, world_size]-sized physical counts array. 
            pca_cumsum_ptr[i, j] = x means x tokens 
        pca_ptr (Tensor): [E, world_size]-sized physical counts array.
            pca_ptr[i, j] = x represents that x tokens are routed from device j
            to expert i on the current rank.
        token_buffer_ptr (Tensor): [E, world_size, capacity, hidden_dim]-sized tensor.
            the output buffer that the all-to-all writes to.
        token_sync_ptr (Tensor): [E]-sized tensor. These are synchronization variables
            set by the prior all-to-all to ensure correctness.
        E (int): number of *local* experts.
        world_size (int): number of participating ranks.
        mxa (int): maximum capaicty (2nd dimension of the token_buffer_ptr array).
        hidden_dim (int): token hidden-dimensions.
    """
    expert = tl.program_id(0)
    dev_id = tl.program_id(1)
    token_id = tl.program_id(2)

    ## We predicate some blocks off immediately. ##
    num_tokens = tl.load(pca_ptr + (expert * world_size + dev_id).to(tl.int64))

    if num_tokens < token_id:
        return ## Immediately terminate.

    ## Else, we have a non-zero token to shift to the output buffer. ##

    ## We have to wait on the prior shmem puts to finish successfuly. 
    while tl.load(token_sync_ptr + expert.to(tl.int64), volatile=True) < world_size:
        pass 
    
    ## We loop over the hidden dimension and shift a token over to the output buffer. 
    inp_ptrs = expert * world_size * mxa * hidden_dim + dev_id *  mxa * hidden_dim + token_id * hidden_dim + tl.arange(BLOCK_X)
    cum_summed_prev = tl.load(pca_cumsum_ptr + (expert * world_size + dev_id).to(tl.int64))
    packed_ptrs = (cum_summed_prev + token_id) * hidden_dim
    for _ in tl.range(tl.cdiv(hidden_dim, BLOCK_X)):

        tkns = tl.load(token_buffer_ptr + inp_ptrs, mask=0)
        tl.store(output_buffer_ptr + packed_ptrs, tkns)

        packed_ptrs += BLOCK_X
        inp_ptrs += BLOCK_X

@triton.jit
def grouped_gemm(
    # device tensor of matrices pointers
    token_ptrs, # Shape: [S, hidden_dim].
    expert_weights, # Shape: [hidden_dim, expert_hidden_dim]
    output_ptrs, # Shape: [S, expert_hidden_dim]
    expert_tkn_cnt_ptr, # Shape: [E], token count per expert.
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # number of gemms -> equivalent to local expert count.
    expert_cnt: tl.contexpr,
    hidden_dim: tl.constexpr,
    expert_hidden_dim: tl.contexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    This kernel implements a grouped-gemm on the input token-buffers.

    Args:
        token_ptrs (Tensor): [S, hidden_dim]-sized array. S is the packed
            number of tokens (no zero-padding) post data-shuffling.
        expert_weights (Tensor): [hidden_dim, expert_hidden_dim]-sized array. 
            This represents each experts' weights.
        output_ptrs (Tensor): [S, expert_hidden_dim]-sized array. Buffer to store
            the results of processing the input tokens with expert weights.
        expert_tkn_cnt_ptr (Tensor): [E]-sized array representing the tokens routed 
            to expert i on the current rank. 
        
        Rest of the arguments are self-explanatory.
    """
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(expert_cnt):
        # get the gemm size of the current problem
        gm = tl.load(expert_tkn_cnt_ptr + g)
        gn = expert_hidden_dim
        gk = hidden_dim
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = token_ptrs + offs_am[:, None] * hidden_dim + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * expert_hidden_dim + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                # assume full tile for now
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(output_ptrs.dtype.element_ty)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + expert_hidden_dim * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def run_counts_and_tokens_exchange(
    #*,
    rank: int,
    world_size: int,
    send_payload: torch.Tensor,  # [sum_send, H]
    send_counts: torch.Tensor,  # [world, E] int32
    dst_offsets: torch.Tensor,  # [world] int32
    buffers: ShmemBuffers,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    stream_comm: Optional[torch.cuda.Stream] = None,
    # Two options for sync variables:
    #  - clear_local_counters=True: zero counters every iteration (simple)
    #  - clear_local_counters=False: use a monotonic epoch and wait for (iter_idx+1)*world_size
    clear_local_counters: bool = True,
    iter_idx: int = 0,
) -> ShmemBuffers:
    """
    Execute Step-1 and Step-2 kernels on the provided comm stream.
    """

    _assert_cuda_int32(send_counts, "send_counts")
    _assert_cuda_int32(dst_offsets, "dst_offsets")

    if send_counts.shape != (world_size, e_local):
        raise ValueError(
            f"send_counts must have shape [world={world_size}, E={e_local}], got {tuple(send_counts.shape)}"
        )

    # Capacity is a fixed upper bound (padded buffer contract).
    max_pair = int(send_counts.max().item()) if send_counts.numel() else 0
    if max_pair > capacity:
        raise ValueError(
            f"CAP too small: max send_counts[dst,e] on rank {rank} is {max_pair}, but CAP={capacity}. "
            "Increase CAP to satisfy the padded buffer contract."
        )

    expert_offs = build_expert_offsets(send_counts)

    # Decide the expected value for the receiver-side counters.
    expected = world_size if clear_local_counters else (iter_idx + 1) * world_size

    # CHANGED: Defensive coding: require comm stream, remove predication / duplicated branches.
    if stream_comm is None:
        raise ValueError(
            "stream_comm must be provided. We always run comm kernels on a dedicated comm stream."
        )

    # CHANGED: Single stream region, no duplicated code.
    with torch.cuda.stream(stream_comm):
        # Reset receiver-side counters if using the simple per-iter reset mode.
        if clear_local_counters:
            buffers.counts_ready.zero_()
            buffers.token_sync.zero_()

    assert stream_comm, 'Incorrectly initialized stream_comm'
    with torch.cuda.stream(stream_comm):
        # Step-1: exchange token counts.
        counts_exchange_kernel[(world_size,)](
            send_counts,
            buffers.pca,
            buffers.counts_ready,
            buffers.heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_E=128,
            num_warps=4,
        )

        # Step-2: exchange tokens.
        tokens_exchange_kernel[(world_size, e_local)](
            buffers.counts_ready,
            send_payload,
            send_counts,
            dst_offsets,
            expert_offs,
            buffers.token_buf,
            buffers.token_sync,
            buffers.heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            CAP=capacity,
            hidden_dim=hidden_dim,
            EXPECTED=expected,
            BLOCK_M=32,
            BLOCK_K=128,
            num_warps=8,
        )

    return buffers
