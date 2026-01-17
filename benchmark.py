

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import triton
import iris
# Do moved some logic to A2A kernel file maybe to make it more clear not only for read but also for tune

from utils import (
    set_seed,
    gen_local_tokens,
    gen_router,
    gen_local_weights,
    route_and_pack_padding_free,
)

# Reuse Triton kernels from the existing overlap implementation.
from alltoall_persistent_gemm import (
    iris_alltoallv_put_kernel,
    iris_alltoallv_signal_kernel,
    iris_overlapped_grouped_gemm_kernel_2d,
)



# Small utilities


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    return int(v) if v is not None else int(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, None)
    return v if v is not None else default


def _init_iris_shmem_compat():
   
  

    sm = getattr(iris, "symmetric_memory", None)
    if sm is not None:
        if hasattr(sm, "SymmetricMemoryProcessGroup"):
            return sm.SymmetricMemoryProcessGroup("WORLD")
        if hasattr(sm, "SymmetricMemory"):
            return sm.SymmetricMemory("WORLD")
        if hasattr(sm, "ProcessGroup"):
            return sm.ProcessGroup("WORLD")
        if hasattr(sm, "enable"):
            return sm.enable("WORLD")

    # Older constructor style
    ctor = getattr(iris, "iris", None)
    if callable(ctor):
        heap_gb = float(os.getenv("IRIS_HEAP_GB", "16"))
        heap_bytes = int(heap_gb * (2**30))
        return ctor(heap_bytes)

    raise RuntimeError(
        "Unable to initialize IRIS symmetric memory. "
        "Set up iris.symmetric_memory, or provide iris.iris(heap_bytes)."
    )


def _assert_int32_cuda(x: torch.Tensor, name: str) -> None:
    assert x.is_cuda, f"{name} must be CUDA"
    assert x.dtype == torch.int32, f"{name} must be int32"



# Metadata stage: counts A2A + offset exchange


def metadata_exchange_counts_and_offsets(
    *,
    send_counts: torch.Tensor,  # [world, E_local] int32
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Implements the *metadata* stage using two synchronous collectives.

    First all-to-all :
      - Each src sends to each dst a vector of length E_local:
            send_counts[dst, :]
      - Each dst receives from each src:
            recv_counts[src, :]  (tokens src sends to this dst's local experts)

    Offset exchange (helper to avoid needing global counts_all):
      - Each dst computes base offsets for each src in its receive buffer
      - Then all-to-all those offsets so each src knows where to write at each dst

    Returns:
      recv_counts:  [world, E_local] int32   (per-src counts for this dst)
      write_offsets: [world] int32          (per-dst base offset for this src)
      total_recv: int                       (exact rows this dst receives)
      total_recv_max: int                   (max(total_recv) across ranks; symmetric)
    """
    _assert_int32_cuda(send_counts, "send_counts")

    world_size, e_local = send_counts.shape

    # First all-to-all: counts 
    send_flat = send_counts.contiguous().view(-1)  # world*E
    recv_flat = torch.empty_like(send_flat)

    # Equal split sizes: each rank sends E_local ints to each dst.
    dist.all_to_all_single(recv_flat, send_flat)
    recv_counts = recv_flat.view(world_size, e_local).contiguous()  # [src, E]

    #  Compute per-src total rows and local base offsets in recv buffer
    recv_src_sizes = recv_counts.to(torch.int64).sum(dim=1)  # [src]
    total_recv = int(recv_src_sizes.sum().item())

    # Base offset for each src block in the dst's recv buffer.
    recv_offsets = (torch.cumsum(recv_src_sizes, dim=0) - recv_src_sizes).to(torch.int32).contiguous()

    # Offset exchange (src learns where to write at each dst) 
    # Each dst sends one int32 offset to each src => all_to_all_single over length world.
    send_off = recv_offsets
    recv_off = torch.empty_like(send_off)
    dist.all_to_all_single(recv_off, send_off)
    write_offsets = recv_off.contiguous()  # [dst]

    # Symmetric heap constraint: compute max(total_recv) across ranks ----ce, dtype=torch.int64)
    dist.all_reduce(total_recv_t, op=dist.ReduceOp.MAX)
    total_recv_max = int(total_recv_t.item())

    return recv_counts, write_offsets, total_recv, total_recv_max



# Reference correctness (NCCL all_to_all_single + serial GEMM)

def run_reference_nccl(
    *,
    world_size: int,
    send_payload: torch.Tensor,  # [sum_send, H]
    send_counts: torch.Tensor,   # [world, E_local] int32
    recv_counts: torch.Tensor,   # [world, E_local] int32
    local_weights: torch.Tensor, # [E_local, H, O]
) -> torch.Tensor:
    """Reference pipeline: true alltoallv + GEMM after comm completes.

    Output layout matches the overlapped consumer kernel:
      rows are packed src-major, then expert-major within each src block.
    """
    _assert_int32_cuda(send_counts, "send_counts")
    _assert_int32_cuda(recv_counts, "recv_counts")

    hidden_dim = int(send_payload.shape[1])
    out_dim = int(local_weights.shape[2])
    e_local = int(local_weights.shape[0])

    input_split_sizes = send_counts.sum(dim=1).to(torch.int64).tolist()  # per dst
    output_split_sizes = recv_counts.sum(dim=1).to(torch.int64).tolist()  # per src
    total_recv = int(sum(output_split_sizes))

    recv_payload = torch.empty((total_recv, hidden_dim), device=send_payload.device, dtype=send_payload.dtype)

    # Second all-to-all (payload) in the senior's pipeline.
    dist.all_to_all_single(
        recv_payload,
        send_payload,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    # Offsets for expert slices within each src block.
    recv_counts_i64 = recv_counts.to(torch.int64)
    expert_offsets = torch.cumsum(recv_counts_i64, dim=1) - recv_counts_i64  # [src, e]
    src_sizes = recv_counts_i64.sum(dim=1)
    src_base = torch.cumsum(src_sizes, dim=0) - src_sizes  # [src]

    out = torch.zeros((e_local, total_recv, out_dim), device=send_payload.device, dtype=send_payload.dtype)

    # Correctness-first reference: Python loops.
    for src in range(world_size):
        sbase = int(src_base[src].item())
        for e in range(e_local):
            n = int(recv_counts_i64[src, e].item())
            if n == 0:
                continue
            off = int(expert_offsets[src, e].item())
            rows = slice(sbase + off, sbase + off + n)
            out[e, rows, :] = recv_payload[rows, :] @ local_weights[e, :, :]

    return out



# Custom overlap step (IRIS puts + atomic signals + overlapped GEMM)


@dataclass
class StepTimings:
    a2a_ms: float
    gemm_ms: float
    e2e_ms: float


def run_custom_overlap_one_iter(
    *,
    rank: int,
    world_size: int,
    send_payload: torch.Tensor,      # [sum_send, H]
    send_counts: torch.Tensor,       # [world, E_local] int32
    recv_counts: torch.Tensor,       # [world, E_local] int32
    write_offsets: torch.Tensor,     # [world] int32: base offset into each dst's recv buffer for this src
    total_recv: int,
    total_recv_max: int,
    local_weights: torch.Tensor,     # [E_local, H, O]
    shmem,
    recv_signal_symm: torch.Tensor,  # [world] int32 (symmetric)
    s_comm: torch.cuda.Stream,
    s_comp: torch.cuda.Stream,
    expected_signal: int,
    profile: bool,
) -> Tuple[torch.Tensor, StepTimings]:
    """One iteration of payload+overlapped GEMM. Metadata is handled outside."""

    _assert_int32_cuda(send_counts, "send_counts")
    _assert_int32_cuda(recv_counts, "recv_counts")
    _assert_int32_cuda(write_offsets, "write_offsets")

    device = send_payload.device
    hidden_dim = int(send_payload.shape[1])
    out_dim = int(local_weights.shape[2])
    e_local = int(local_weights.shape[0])

    # Receiver-side offsets for the consumer kernel.
    recv_counts_i64 = recv_counts.to(torch.int64)
    recv_expert_offs = (torch.cumsum(recv_counts_i64, dim=1) - recv_counts_i64).to(torch.int32).contiguous()
    src_sizes = recv_counts_i64.sum(dim=1)
    recv_src_base = (torch.cumsum(src_sizes, dim=0) - src_sizes).to(torch.int32).contiguous()

    # Sender-side sizes/offsets into packed send_payload.
    send_dst_sizes = send_counts.sum(dim=1).to(torch.int32).contiguous()  # [dst]
    send_dst_offsets = (torch.cumsum(send_dst_sizes.to(torch.int64), dim=0) - send_dst_sizes.to(torch.int64)).to(torch.int32).contiguous()

    # Dynamic symmetric payload buffer (Ahan's requirement).
    recv_payload_symm = shmem.zeros((int(total_recv_max), hidden_dim), dtype=send_payload.dtype, device="cuda")

    # Output buffer: exact total_recv rows for this destination.
    out = torch.zeros((e_local, int(total_recv), out_dim), device=device, dtype=send_payload.dtype)

    if profile:
        ev_a2a_s = torch.cuda.Event(enable_timing=True)
        ev_a2a_e = torch.cuda.Event(enable_timing=True)
        ev_g_s = torch.cuda.Event(enable_timing=True)
        ev_g_e = torch.cuda.Event(enable_timing=True)
        ev_e_s = torch.cuda.Event(enable_timing=True)
        ev_e_e = torch.cuda.Event(enable_timing=True)
        ev_e_s.record()

    # COMM stream: IRIS puts + remote signal 
        if profile:
            ev_a2a_s.record()

        rows_max = int(send_dst_sizes.max().item()) if send_dst_sizes.numel() else 0
        if rows_max > 0:
            grid_put = (world_size, triton.cdiv(rows_max, 128), 1)
            iris_alltoallv_put_kernel[grid_put](
                send_payload,
                recv_payload_symm,
                send_dst_offsets,
                send_dst_sizes,
                write_offsets,               # dst_src_prefix_ptr
                shmem.get_heap_bases(),
                hidden_dim=hidden_dim,
                src_rank=rank,
                world_size=world_size,
                BLOCK_M=128,
                BLOCK_K=256,
                num_warps=8,
            )

        # Always signal (even if send_dst_sizes[dst]==0) to avoid deadlocks.
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

    # Compute stream: overlapped grouped GEMM (waits per-src) 
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
                total_recv=int(total_recv),
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

    # Per-iteration boundary: default stream waits for both streams.
    torch.cuda.current_stream().wait_stream(s_comm)
    torch.cuda.current_stream().wait_stream(s_comp)

    if profile:
        ev_e_e.record()
        ev_e_e.synchronize()
        timings = StepTimings(
            a2a_ms=ev_a2a_s.elapsed_time(ev_a2a_e),
            gemm_ms=ev_g_s.elapsed_time(ev_g_e),
            e2e_ms=ev_e_s.elapsed_time(ev_e_e),
        )
    else:
        torch.cuda.synchronize()
        timings = StepTimings(0.0, 0.0, 0.0)

    return out, timings


# Correctness helper


def correctness_check(a: torch.Tensor, b: torch.Tensor, *, rtol=1e-2, atol=1e-2) -> None:
    # BF16 can be noisy; keep tolerances modest.
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)



# mp.spawn callee


def callee(
    rank: int,
    world_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    topk: int,
    e_local: int,
    base_seed: int,
    iters: int,
    warmup: int,
    do_correctness: bool,
) -> None:
    #  device mapping (single-node mp.spawn harness) 
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    # distributed init (single-node)
    master_addr = _env_str("MASTER_ADDR", "127.0.0.1")
    master_port = _env_str("MASTER_PORT", "29500")
    init_method = f"tcp://{master_addr}:{master_port}"

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )

    #  reproducibility
    set_seed(base_seed, rank)

    dtype = torch.bfloat16
    out_dim = hidden_dim
    num_experts_total = world_size * e_local

    # generate tokens + routing
    tokens = gen_local_tokens(
        batch=batch,
        seq=seq,
        hidden_dim=hidden_dim,
        dtype=dtype,
        device=device,
        base_seed=base_seed,
        rank=rank,
    )
    router = gen_router(
        hidden_dim=hidden_dim,
        num_experts=num_experts_total,
        dtype=dtype,
        device=device,
        base_seed=base_seed,
    )

    send_payload, send_counts, _, _ = route_and_pack_padding_free(
        tokens=tokens,
        router=router,
        topk=topk,
        world_size=world_size,
        num_experts_total=num_experts_total,
    )

    # weights (local experts only) 
    local_weights = gen_local_weights(
        e_local=e_local,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dtype=dtype,
        device=device,
        base_seed=base_seed,
        rank=rank,
    )

    #shmem / signal / streams (persistent)
    shmem = _init_iris_shmem_compat()
    recv_signal_symm = shmem.zeros((world_size,), dtype=torch.int32, device="cuda")
    s_comm = torch.cuda.Stream()
    s_comp = torch.cuda.Stream()

    expected_signal = 0

    # Correctness check (single iter)

    if do_correctness:
        dist.barrier()
        torch.cuda.synchronize()

        recv_counts, write_offsets, total_recv, total_recv_max = metadata_exchange_counts_and_offsets(
            send_counts=send_counts
        )

        expected_signal += 1
        out_custom, _ = run_custom_overlap_one_iter(
            rank=rank,
            world_size=world_size,
            send_payload=send_payload,
            send_counts=send_counts,
            recv_counts=recv_counts,
            write_offsets=write_offsets,
            total_recv=total_recv,
            total_recv_max=total_recv_max,
            local_weights=local_weights,
            shmem=shmem,
            recv_signal_symm=recv_signal_symm,
            s_comm=s_comm,
            s_comp=s_comp,
            expected_signal=expected_signal,
            profile=False,
        )

        out_ref = run_reference_nccl(
            world_size=world_size,
            send_payload=send_payload,
            send_counts=send_counts,
            recv_counts=recv_counts,
            local_weights=local_weights,
        )

        correctness_check(out_custom, out_ref)

        if rank == 0:
            print("[OK] correctness_check passed")

    # Warmup (not timed)
 
    dist.barrier()
    for _ in range(warmup):
        recv_counts, write_offsets, total_recv, total_recv_max = metadata_exchange_counts_and_offsets(
            send_counts=send_counts
        )
        expected_signal += 1
        _out, _t = run_custom_overlap_one_iter(
            rank=rank,
            world_size=world_size,
            send_payload=send_payload,
            send_counts=send_counts,
            recv_counts=recv_counts,
            write_offsets=write_offsets,
            total_recv=total_recv,
            total_recv_max=total_recv_max,
            local_weights=local_weights,
            shmem=shmem,
            recv_signal_symm=recv_signal_symm,
            s_comm=s_comm,
            s_comp=s_comp,
            expected_signal=expected_signal,
            profile=False,
        )
    dist.barrier()


    # Timed iterations

    a2a_ms = gemm_ms = e2e_ms = 0.0

    for _ in range(iters):
        recv_counts, write_offsets, total_recv, total_recv_max = metadata_exchange_counts_and_offsets(
            send_counts=send_counts
        )
        expected_signal += 1
        _out, t = run_custom_overlap_one_iter(
            rank=rank,
            world_size=world_size,
            send_payload=send_payload,
            send_counts=send_counts,
            recv_counts=recv_counts,
            write_offsets=write_offsets,
            total_recv=total_recv,
            total_recv_max=total_recv_max,
            local_weights=local_weights,
            shmem=shmem,
            recv_signal_symm=recv_signal_symm,
            s_comm=s_comm,
            s_comp=s_comp,
            expected_signal=expected_signal,
            profile=True,
        )
        a2a_ms += t.a2a_ms
        gemm_ms += t.gemm_ms
        e2e_ms += t.e2e_ms

    # Use a max-reduction to report worst-rank time (HPC convention).
    avg = torch.tensor([a2a_ms / iters, gemm_ms / iters, e2e_ms / iters], device=device, dtype=torch.float32)
    dist.all_reduce(avg, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"avg_iter_ms(max_rank): {avg[2].item():.3f} | a2a_ms: {avg[0].item():.3f} | gemm_ms: {avg[1].item():.3f} "
            f"(iters={iters}, warmup={warmup}, batch={batch}, seq={seq}, hidden={hidden_dim}, topk={topk}, E_local={e_local})"
        )

    dist.destroy_process_group()





if __name__ == "__main__":
    # Keep defaults close to your existing harness; all overridable via env.
    world_size = _env_int("WORLD_SIZE", 8)
    batch = _env_int("BATCH", 4)
    seq = _env_int("SEQ", 1024)
    hidden_dim = _env_int("HIDDEN", 768)
    topk = _env_int("TOPK", 2)
    e_local = _env_int("ELOCAL", 32)
    base_seed = _env_int("SEED", 42)

    iters = _env_int("ITERS", 30)
    warmup = _env_int("WARMUP", 10)
    do_correctness = bool(_env_int("DO_CORRECTNESS", 1))

    mp.spawn(
        callee,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, base_seed, iters, warmup, do_correctness),
        nprocs=world_size,
        join=True,
    )
