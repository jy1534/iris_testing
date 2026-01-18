from __future__ import annotations


from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

import iris


"""
world = DeviceCount E = E_local = T 
H = hidden_dimension 
CAP = max_{e,s} pca[e,s] 
Capcity in real moe 
"""


#safety helpers
# in case counts not int 32 and sync tensor not in shmem

def _assert_cuda_int32(x: torch.Tensor, name: str) -> None:
    
    assert x.is_cuda, f"{name} must be CUDA"
    assert x.dtype == torch.int32, f"{name} must be int32"


@dataclass
class ShmemStep12Buffers:
    # Step-1 outputs / sync
    pca: torch.Tensor  # [E, world] int32 (symmetric)
    counts_ready: torch.Tensor  # [1] int32 (symmetric)

    # Step-2 outputs / sync
    token_buf: torch.Tensor  # [E, world, CAP, H] (symmetric)
    token_sync: torch.Tensor  # [E] int32 (symmetric)

    # Cached heap bases (IRIS addressing)
    heap_bases: torch.Tensor


def init_step12_buffers(
    *,
    shmem,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    token_dtype: torch.dtype,
) -> ShmemStep12Buffers:
    """
    Allocate symmetric buffers with fixed shapes.
    """

    # pca[e, src] = counts for this device's local expert e sent by src
    pca = shmem.zeros((e_local, world_size), dtype=torch.int32, device="cuda")

    # counts_ready becomes == world_size once all senders finished writing counts.
    counts_ready = shmem.zeros((1,), dtype=torch.int32, device="cuda")

    # token_buf[e, src, m, :] (m in [0, CAP)) holds tokens from src for expert e.
    token_buf = shmem.zeros((e_local, world_size, capacity, hidden_dim), dtype=token_dtype, device="cuda")

    # token_sync[e] becomes == world_size once all senders finished sending tokens for expert e.
    token_sync = shmem.zeros((e_local,), dtype=torch.int32, device="cuda")

    heap_bases = shmem.get_heap_bases()

    return ShmemStep12Buffers(
        pca=pca,
        counts_ready=counts_ready,
        token_buf=token_buf,
        token_sync=token_sync,
        heap_bases=heap_bases,
    )

# Step-1 kernel: counts exchange
@triton.jit
def iris_counts_exchange_kernel(
    send_counts_ptr,  # [world, E] int32 (local)
    pca_ptr,  # [E, world] int32 (symmetric on dst)
    counts_ready_ptr,  # [1] int32 (symmetric on dst)
    heap_bases,
    *,
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

# Step-2 kernel: token exchange (with local spin-wait on counts_ready) which in the run may cause busy wait right
@triton.jit
def iris_tokens_exchange_kernel(
    counts_ready_ptr,      # [1] int32 local
    send_ptr,
    send_counts_ptr,
    dst_offsets_ptr,
    expert_offs_ptr,
    token_buf_ptr,
    token_sync_ptr,
    heap_bases,
    *,
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

## TODO(ahanugupta): investigate if we should remove mxa as tl.constexpr. 
## This will change between iterations and trigger recompilation.
@triton.jit
def token_shuffle(
    pca_cumsum_ptr, pca_ptr, ## Both of size: [E, world_size]
    token_buffer_ptr, # Size: [E, world_size, mxa, hidden_dim]
    output_buffer_ptr, # Size: [S, hidden_dim]
    E: tl.constexpr, world_size: tl.constexpr,
    mxa: tl.constexpr, hidden_dim: tl.constexpr
    BLOCK_X: tl.contexpr  ## We have 1-d blocks only.
):
    expert = tl.program_id(0)
    dev_id = tl.program_id(1)
    token_id = tl.program_id(2)

    ## We predicate some blocks off immediately. ##
    num_tokens = tl.load(pca_ptr + (expert * world_size + dev_id).to(tl.int64))

    if num_tokens < token_id:
        return ## Immediately terminate.

    ## Else, we have a non-zero token to shift to the output buffer. ##
    
    ## We loop over the hidden dimension and shift a token over to the output buffer. 
    inp_ptrs = expert * world_size * mxa * hidden_dim + dev_id *  mxa * hidden_dim + token_id * hidden_dim + tl.arange(BLOCK_X)
    cum_summed_prev = tl.load(pca_cumsum_ptr + (expert * world_size + dev_id).to(tl.int64))
    packed_ptrs = (cum_summed_prev + token_id) * hidden_dim
    for _ in tl.range(tl.cdiv(hidden_dim, BLOCK_X)):

        tkns = tl.load(token_buffer_ptr + inp_ptrs, mask=0)
        tl.store(output_buffer_ptr + packed_ptrs, tkns)

        packed_ptrs += BLOCK_X
        inp_ptrs += BLOCK_X

def step4_grouped_gemm_triton(*args, **kwargs):
    raise NotImplementedError("Step 4 (grouped GEMM with per-expert spin-wait) is not implemented in this file.")

# Step-1/2 run wrapper
def build_expert_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """
    Prefix offsets within each destination block in send_payload.

    send_counts: [world, E] int32
    returns: expert_offs[dst, e] (row offset within the dst block).
    """

    _assert_cuda_int32(send_counts, "send_counts")
    sc64 = send_counts.to(torch.int64)
    offs = (torch.cumsum(sc64, dim=1) - sc64).to(torch.int32)
    return offs.contiguous()


def run_step12(
    *,
    rank: int,
    world_size: int,
    send_payload: torch.Tensor,  # [sum_send, H]
    send_counts: torch.Tensor,  # [world, E] int32
    dst_offsets: torch.Tensor,  # [world] int32
    buffers: ShmemStep12Buffers,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    stream_comm: Optional[torch.cuda.Stream] = None,
    # Two options for sync variables:
    #  - clear_local_counters=True: zero counters every iteration (simple)
    #  - clear_local_counters=False: use a monotonic epoch and wait for (iter_idx+1)*world_size
    clear_local_counters: bool = True,
    iter_idx: int = 0,
) -> ShmemStep12Buffers:
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

    # Reset receiver-side counters if using the simple per-iter reset mode.
    if clear_local_counters:
        if stream_comm is None:
            buffers.counts_ready.zero_()
            buffers.token_sync.zero_()
        else:
            with torch.cuda.stream(stream_comm):
                buffers.counts_ready.zero_()
                buffers.token_sync.zero_()

    assert stream_comm, 'Incorrectly initialized stream_comm'
    with torch.cuda.stream(stream_comm):
        # Step-1: exchange token counts.
        iris_counts_exchange_kernel[(world_size,)](
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
        iris_tokens_exchange_kernel[(world_size, e_local)](
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

    ## First, we aggregrate token counts and create a packed output buffer.
    cum_summed_tkn_cnt = torch.roll(buffers.pca.view(-1).cumsum(), shifts=1)
    total_tkn_cnt = cum_summed_tkn_cnt[-1]
    output_buffer = torch.zeros((total_tkn_cnt, hidden_dim), dtype=send_payload.dtype).to(send_payload.device)
    cum_summed_tkn_cnt[0] = 0
    cum_summed_tkn_cnt = cum_summed_tkn_cnt.view(e_local, -1)
    ## Next, launch token shuffling kernel. ##
    grid = (e_local, world_size, buffers.token_buf.size(2))

    token_shuffle[grid](
        cum_summed_tkn_cnt, buffers.pca,
        buffers.token_buf, 
        output_buffer,
        e_local, world_size,
        buffers.token_buf.size(2), hidden_dim,
        128
    )

    return buffers


