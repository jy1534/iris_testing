

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

import triton
import triton.language as tl

import iris



# Payload transfer: IRIS put kernels (padding-free per pair)

@triton.jit
def iris_alltoallv_put_kernel(
    send_ptr,                    # [sum_send, H]
    recv_ptr,                    # [recv_max, H] (symmetric heap)
    send_dst_offsets_ptr,        # [world] int32 (prefix in send_ptr)
    send_dst_sizes_ptr,          # [world] int32 (#rows to each dst)
    dst_src_prefix_ptr,          # [world] int32 (offset into dst's recv buffer for this src)
    heap_bases,
    hidden_dim: tl.constexpr,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Copy payload rows to each destination using IRIS puts.

    Grid:
      - pid0: destination rank in [0, world)
      - pid1: tile index over rows (M dimension)

    avoid doing any signaling here. Signaling is done by a separate
    kernel launched *after* this kernel on the same comm stream.

    --- maybe modification points ---
    [MOD-A2A-1]  BLOCK_M/BLOCK_K and num_warps.
    [MOD-A2A-2]  If IRIS offers `quiet()` / completion semantics to add before signaling for stricter ordering.
    """

    dst = tl.program_id(0)
    pid_m = tl.program_id(1)

    # how many rows we send to this destination
    send_rows = tl.load(send_dst_sizes_ptr + dst).to(tl.int32)
    if send_rows == 0:
        return

    send_base = tl.load(send_dst_offsets_ptr + dst).to(tl.int32)

    # offset into destination's recv buffer where *this src* should write
    dst_base = tl.load(dst_src_prefix_ptr + dst).to(tl.int32)

    # row indices within this dst message
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < send_rows

    # pointers for row-major [row, col]
    row_ids = (send_base + offs_m).to(tl.int32)

    # loop over hidden dim tiles
    for k in tl.static_range(0, tl.cdiv(hidden_dim, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_dim

        # 2D pointers: [M, K]
        send_ptrs = send_ptr + row_ids[:, None] * hidden_dim + offs_k[None, :]
        recv_ptrs = recv_ptr + (dst_base + offs_m)[:, None] * hidden_dim + offs_k[None, :]

        x = tl.load(send_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Remote write to destination `dst`.
        iris.put(
            recv_ptrs,
            x,
            src_rank,
            dst,
            heap_bases,
            mask=m_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def iris_alltoallv_signal_kernel(
    signal_ptr,                 # [world] int32 on destination (symmetric heap)
    send_dst_sizes_ptr,         # [world] int32
    heap_bases,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """After all puts are complete (stream order), send a per-destination ready signal."""
    dst = tl.program_id(0)
    """
    Even if we send 0 rows to a destination this iteration, we *must* still send a completion signal. Otherwise, the destination's compute kernel
    will spin forever waiting for signal[src_rank] to reach expected_signal.


    signal_ptr lives in symmetric heap; iris.atomic_add writes to remote dst.
    """
    iris.atomic_add(
        signal_ptr + src_rank,
        1,
        src_rank,
        dst,
        heap_bases,
        sem="release",
        scope="sys",
    )



#  Consumer: grouped GEMM that waits per-src (true overlap)


@triton.jit
def iris_overlapped_grouped_gemm_kernel(
    recv_ptr,                   # [recv_max, H] (local view; already filled by IRIS)
    signal_ptr,                 # [world] int32 (local view)
    w_ptr,                      # [E, H, O]
    out_ptr,                    # [E, total_recv, O]
    recv_counts_ptr,            # [world, E] int32
    recv_expert_offs_ptr,        # [world, E] int32
    recv_src_base_ptr,           # [world] int32
    total_recv: tl.constexpr,
    hidden_dim: tl.constexpr,
    out_dim: tl.constexpr,
    expected_signal: tl.constexpr,
    # strides
    stride_w_e, stride_w_h, stride_w_o,
    stride_out_e, stride_out_m, stride_out_o,
    stride_c_src, stride_c_e,
    stride_off_src, stride_off_e,
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-(src,expert) GEMM tile; waits on signal[src] then computes.

    Grid:
      - pid0: pid_m (tile in M for this expert slice)
      - pid1: expert id
      - pid2: src rank

    --- Potential modification points ---
    [MOD-GEMM-1] Autotune BLOCK_M/BLOCK_N/BLOCK_K + num_warps.
    [MOD-GEMM-2] Use fp16/bf16 accumulation settings, tl.multiple_of, tl.max_contiguous.
    [MOD-GEMM-3] Replace spin-wait
    """

    pid_m = tl.program_id(0)
    expert = tl.program_id(1)
    src = tl.program_id(2)

    # wait until src has signaled completion for this iteration
    # volatile=True prevents compiler from caching.
    sig_addr = signal_ptr + src
    while tl.load(sig_addr, volatile=True) < expected_signal:
        pass

    # valid tokens for this (src, expert)
    n = tl.load(recv_counts_ptr + src * stride_c_src + expert * stride_c_e).to(tl.int32)
    if n == 0:
        return

    m_start = pid_m * BLOCK_M
    if m_start >= n:
        return

    src_base = tl.load(recv_src_base_ptr + src).to(tl.int32)
    e_off = tl.load(recv_expert_offs_ptr + src * stride_off_src + expert * stride_off_e).to(tl.int32)
    base = src_base + e_off

    # M/N/K offsets 
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) + 0  # pid_n is fused by launching BLOCK_N tiles in grid? (see wrapper)

    """
      We fuse N dimension into pid_m by mapping pid_m over (m_tile, n_tile) in wrapper.
     So recover pid_n here:
       global_pid_m = pid_m
       num_pid_m = ceil(n / BLOCK_M)
       pid_n = global_pid_m // num_pid_m
       pid_m_local = global_pid_m - pid_n * num_pid_m
     Since n is runtime; doing this inside kernel is messy.
     Instead, wrapper launches pid_m over M only and we use a 2D grid for N.
    => This kernel expects to be launched with axis0 over M tiles and axis3 over N tiles.

     (placeholder; overridden by the actual wrapper kernel below)
    """
    tl.static_assert(False, "Do not launch iris_overlapped_grouped_gemm_kernel directly")


@triton.jit
def iris_overlapped_grouped_gemm_kernel_2d(
    recv_ptr,                   # [recv_max, H]
    signal_ptr,                 # [world]
    w_ptr,                      # [E, H, O]
    out_ptr,                    # [E, total_recv, O]
    recv_counts_ptr,            # [world, E]
    recv_expert_offs_ptr,        # [world, E]
    recv_src_base_ptr,           # [world]
    total_recv: tl.constexpr,
    hidden_dim: tl.constexpr,
    out_dim: tl.constexpr,
    expected_signal,
    # strides
    stride_w_e, stride_w_h, stride_w_o,
    stride_out_e, stride_out_m, stride_out_o,
    stride_c_src, stride_c_e,
    stride_off_src, stride_off_e,
    # meta
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Same as above, but with explicit N-tiling.

    Grid:
      - pid0: pid_m tile
      - pid1: pid_n tile
      - pid2: expert
      - pid3: src
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    expert = tl.program_id(2)
    src = tl.program_id(3)

    # Fast-path: if this (src, expert) has no tokens this iteration,
    # return without waiting on the signal. This avoids deadlocks and
    # reduces unnecessary spin-wait traffic.
    n = tl.load(recv_counts_ptr + src * stride_c_src + expert * stride_c_e).to(tl.int32)
    if n == 0:
        return

    sig_addr = signal_ptr + src
    while tl.load(sig_addr, volatile=True) < expected_signal:
        pass

    m_start = pid_m * BLOCK_M
    if m_start >= n:
        return

    src_base = tl.load(recv_src_base_ptr + src).to(tl.int32)
    e_off = tl.load(recv_expert_offs_ptr + src * stride_off_src + expert * stride_off_e).to(tl.int32)
    base = src_base + e_off

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < n
    n_mask = offs_n < out_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # base pointers
    # recv is row-major [row, h]
    recv_base = recv_ptr + (base + offs_m)[:, None] * hidden_dim
    w_base = w_ptr + expert * stride_w_e

    for k in tl.static_range(0, tl.cdiv(hidden_dim, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hidden_dim

        a_ptrs = recv_base + offs_k[None, :]
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = w_base + offs_k[:, None] * stride_w_h + offs_n[None, :] * stride_w_o
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(a, b)

    # store
    out_base = out_ptr + expert * stride_out_e
    out_ptrs = out_base + (base + offs_m)[:, None] * stride_out_m + offs_n[None, :] * stride_out_o
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


# Python orchestration


@dataclass
class IrisRunResult:
    out: torch.Tensor
    a2a_ms: float
    gemm_ms: float
    e2e_ms: float


def run(
    *,
    rank: int,
    world_size: int,
    send_payload: torch.Tensor,             # [sum_send, H]
    send_counts: torch.Tensor,              # [world, E_local] int32
    counts_all: torch.Tensor,               # [world, world, E_local] int32
    local_weights: torch.Tensor,            # [E_local, H, O]
    shmem,
    profile: bool = False,
    iters: int = 1,
    expected_signal_start: int = 0,
) -> IrisRunResult:
    """
    Run IRIS payload A2A + overlapped grouped GEMM.

    This assumes `counts_all` is already computed (metadata stage).

    --- Potential modification points ---
    [MOD-PIPE-1] Maybe switch from symmetric `recv_max = max(total_recv)` to a persistent pool.
     """

    assert send_payload.is_cuda and local_weights.is_cuda
    assert send_counts.dtype == torch.int32
    assert counts_all.dtype == torch.int32

    device = send_payload.device
    hidden_dim = send_payload.shape[1]
    out_dim = local_weights.shape[2]
    e_local = local_weights.shape[0]

    # local recv counts for this destination rank 
    recv_counts = counts_all[:, rank, :].contiguous()  # [src, E]
    recv_counts_i64 = recv_counts.to(torch.int64)

    src_sizes = recv_counts_i64.sum(dim=1)                        # [src]
    total_recv = int(src_sizes.sum().item())

    """
    Symmetric heap constraint: every rank must call shmem.zeros() with the
    same shape per-iteration. Therefore take MAX(total_recv) over ranks.
       """
    total_recv_t = torch.tensor([total_recv], device=device, dtype=torch.int64)
    dist.all_reduce(total_recv_t, op=dist.ReduceOp.MAX)
    total_recv_max = int(total_recv_t.item())

    # precompute receiver-side offsets (local tensors) 
    recv_expert_offs = (torch.cumsum(recv_counts_i64, dim=1) - recv_counts_i64).to(torch.int32)
    recv_src_base = (torch.cumsum(src_sizes, dim=0) - src_sizes).to(torch.int32)

    # sender-side per-destination sizes and offsets into send_payload 
    send_dst_sizes = send_counts.sum(dim=1).to(torch.int32).contiguous()   # [dst]
    send_dst_offsets = (torch.cumsum(send_dst_sizes.to(torch.int64), dim=0) - send_dst_sizes.to(torch.int64)).to(torch.int32)

    # destination offsets (where I write into each dst's recv buffer) 
    # dst_src_prefix[dst] = sum_{s < rank} sum_e counts_all[s, dst, e]
    # (needed to avoid an extra offset-exchange collective)
    per_dst_src_sizes = counts_all[:, :, :].to(torch.int64).sum(dim=2)  # [src, dst]
    dst_src_prefix = per_dst_src_sizes[:rank, :].sum(dim=0).to(torch.int32).contiguous()  # [dst]

    # symmetric heap buffers allocated *per iteration* (Ahan's instruction) 
    # - Payload buffer is dynamically reallocated per-iteration .
    # -Signal buffer is persistent within this `run()` call, and we use an epoch counter (expected_signal) to avoid per-iteration zeroing.
    recv_signal_symm = shmem.zeros((world_size,), dtype=torch.int32, device="cuda")

    # Streams (persistent across iters)
    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()

    # profiling events
    if profile:
        ev_a2a_s = torch.cuda.Event(enable_timing=True)
        ev_a2a_e = torch.cuda.Event(enable_timing=True)
        ev_g_s = torch.cuda.Event(enable_timing=True)
        ev_g_e = torch.cuda.Event(enable_timing=True)
        ev_e2e_s = torch.cuda.Event(enable_timing=True)
        ev_e2e_e = torch.cuda.Event(enable_timing=True)
    else:
        ev_a2a_s = ev_a2a_e = ev_g_s = ev_g_e = ev_e2e_s = ev_e2e_e = None

    # NOTE: we implement an epoch counter in signals to avoid needing global barriers.
    expected_signal = int(expected_signal_start)

    # warmup / iters
    a2a_ms = gemm_ms = e2e_ms = 0.0

    for it in range(iters):
        expected_signal += 1

        # End-to-end GPU timing: record start on the default stream
        # before enqueuing work on other streams.
        if profile:
            ev_e2e_s.record()

        # Dynamic payload buffer instantiation per iteration (Ahan's instruction).
        # Shape must be identical across ranks per-iteration for symmetric heap.
        recv_payload_symm = shmem.zeros((total_recv_max, hidden_dim), dtype=send_payload.dtype, device="cuda")

        out = torch.zeros((e_local, total_recv, out_dim), device=device, dtype=send_payload.dtype)

        # COMM STREAM: payload puts + remote signal
        with torch.cuda.stream(s_comm):
            if profile:
                ev_a2a_s.record()

            # put kernel
            rows_max = int(send_dst_sizes.max().item()) if send_dst_sizes.numel() else 0
            if rows_max > 0:
                grid_put = (
                    world_size,
                    triton.cdiv(rows_max, 128),
                    1,
                )

                iris_alltoallv_put_kernel[grid_put](
                    send_payload,
                    recv_payload_symm,
                    send_dst_offsets,
                    send_dst_sizes,
                    dst_src_prefix,
                    shmem.get_heap_bases(),
                    hidden_dim=hidden_dim,
                    src_rank=rank,
                    world_size=world_size,
                    BLOCK_M=128,
                    BLOCK_K=256,
                    num_warps=8,
                )

            # signal kernel (after puts in stream order)
            iris_alltoallv_signal_kernel[(world_size,)](
                recv_signal_symm,
                send_dst_sizes,
                shmem.get_heap_bases(),
                src_rank=rank,
                world_size=world_size,
                num_warps=4,
            )

            if profile:
                ev_a2a_e.record()

        # COMPUTE STREAM: overlapped grouped GEMM 
        with torch.cuda.stream(s_comp):
            if profile:
                ev_g_s.record()

            max_m = int(recv_counts.max().item()) if recv_counts.numel() else 0
            if max_m > 0 and total_recv > 0:
                grid_gemm = (
                    triton.cdiv(max_m, 128),      # pid_m
                    triton.cdiv(out_dim, 128),    # pid_n
                    e_local,                      # expert
                    world_size,                   # src
                )

                iris_overlapped_grouped_gemm_kernel_2d[grid_gemm](
                    recv_payload_symm,
                    recv_signal_symm,
                    local_weights,
                    out,
                    recv_counts,
                    recv_expert_offs,
                    recv_src_base,
                    total_recv=total_recv,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    expected_signal=expected_signal,
                    stride_w_e=local_weights.stride(0),
                    stride_w_h=local_weights.stride(1),
                    stride_w_o=local_weights.stride(2),
                    stride_out_e=out.stride(0),
                    stride_out_m=out.stride(1),
                    stride_out_o=out.stride(2),
                    stride_c_src=recv_counts.stride(0),
                    stride_c_e=recv_counts.stride(1),
                    stride_off_src=recv_expert_offs.stride(0),
                    stride_off_e=recv_expert_offs.stride(1),
                    BLOCK_M=128,
                    BLOCK_N=128,
                    BLOCK_K=32,
                    num_warps=8,
                )

            if profile:
                ev_g_e.record()

        # end-to-end sync (per iteration boundary) 

        # Ensure default stream waits for both streams before recording end.
        torch.cuda.current_stream().wait_stream(s_comm)
        torch.cuda.current_stream().wait_stream(s_comp)

        if profile:
            ev_e2e_e.record()
            ev_e2e_e.synchronize()
            a2a_ms += ev_a2a_s.elapsed_time(ev_a2a_e)
            gemm_ms += ev_g_s.elapsed_time(ev_g_e)
            e2e_ms += ev_e2e_s.elapsed_time(ev_e2e_e)
        else:
            torch.cuda.synchronize()

    if profile and iters > 0:
        a2a_ms /= iters
        gemm_ms /= iters
        e2e_ms /= iters

    return IrisRunResult(out=out, a2a_ms=a2a_ms, gemm_ms=gemm_ms, e2e_ms=e2e_ms)
