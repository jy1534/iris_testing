import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import triton
import triton.language as tl
import iris

from kernels import run_counts_and_tokens_exchange, alloc_shmem_buffers
from baseline import run_baseline_ref, init_baseline_buffers

from utils import set_seed, gen_local_tokens, gen_router, route_and_pack_padding_free


def _init_iris_shmem():
    import iris, os
    heap_gib = int(os.getenv("IRIS_HEAP_GIB", "100"))
    heap_size = heap_gib * (2**30)
    return iris.iris(heap_size)


@triton.jit
def iris_wait_token_sync_kernel(
    token_sync_ptr,
    heap_bases,
    *,
    src_rank: tl.constexpr,
    world_size: tl.constexpr,
    e_local: tl.constexpr,
):
  
    e = tl.program_id(0)
    if e >= e_local:
        return

    # Spin (device-side) until all senders have signaled.
    while tl.load(token_sync_ptr + e, volatile=True).to(tl.int32) < world_size:
        pass

    # Acquire fence: "self" atomic with acquire semantics.
    iris.atomic_add(
        token_sync_ptr + e,
        0,
        src_rank,
        src_rank,
        heap_bases,
        sem="acquire",
        scope="sys",
    )


def _build_dst_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """dst_offsets[dst] = prefix sum of total tokens to earlier destinations."""
    # send_counts: [world, E_local] int32
    send_dst_sizes = send_counts.sum(dim=1).to(torch.int32)
    dst_offsets = (torch.cumsum(send_dst_sizes, dim=0) - send_dst_sizes).to(torch.int32)
    return dst_offsets.contiguous()


def _masked_stats(
    triton_buf: torch.Tensor,
    torch_buf: torch.Tensor,
    counts_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute max|diff| and sums over ONLY the valid (non-padding) region.

    triton_buf/torch_buf: [E, world, CAP, H]
    counts_mat:           [E, world]  (counts_mat[e, src] = number of valid rows)

    Returns: (max_diff, sum_triton, sum_torch) as 0-dim float32 tensors on GPU.
    """
    assert triton_buf.shape == torch_buf.shape
    E, W, CAP, H = triton_buf.shape

    # mask[e, src, m] = (m < counts_mat[e, src])
    m = torch.arange(CAP, device=triton_buf.device, dtype=torch.int32)[None, None, :]
    mask = (m < counts_mat.to(torch.int32)[:, :, None]).unsqueeze(-1)  # [E, W, CAP, 1]

    diff = (triton_buf - torch_buf).abs().to(torch.float32)
    diff_masked = diff * mask.to(torch.float32)

    max_diff = diff_masked.max()

    sum_triton = (triton_buf.to(torch.float32) * mask.to(torch.float32)).sum()
    sum_torch = (torch_buf.to(torch.float32) * mask.to(torch.float32)).sum()

    return max_diff, sum_triton, sum_torch


def check_compare(
    rank: int,
    world_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    topk: int,
    e_local: int,
    capacity: int,
    base_seed: int,
):
    # init process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    set_seed(base_seed, rank)

    # generate payload + routing
    num_experts_total = world_size * e_local
    tokens = gen_local_tokens(batch, seq, hidden_dim, torch.bfloat16, device, base_seed, rank)
    router = gen_router(hidden_dim, num_experts_total, torch.bfloat16, device, base_seed)

    send_payload, send_counts, _, _ = route_and_pack_padding_free(
        tokens, router, topk, world_size, num_experts_total
    )
    send_counts = send_counts.to(torch.int32).contiguous()  # [world, E_local]

    # gather global counts for baseline & expected PCA
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src, dst, E_local]

    # expected PCA on this rank: pca[e, src] = counts_all[src, rank, e]
    expected_pca = counts_all[:, rank, :].transpose(0, 1).contiguous()  # [E_local, world]

    # init shmem buffers
    shmem = _init_iris_shmem()
    buffers = alloc_shmem_buffers(
    shmem=shmem,
    world_size=world_size,
    e_local=e_local,
    capacity=capacity,
    hidden_dim=hidden_dim,
    token_dtype=torch.bfloat16,
)

    # clear token_buf for repeated runs (slow but removes any ambiguity)
    if os.getenv("CLEAR_TOKEN_BUF", "0") == "1":
        buffers.token_buf.zero_()

    dist.barrier()

    dst_offsets = _build_dst_offsets(send_counts)

    comm_stream = torch.cuda.Stream(device=device)
    # run custom step1/2
    run_counts_and_tokens_exchange(
        rank=rank,
        world_size=world_size,
        send_payload=send_payload,
        send_counts=send_counts,
        dst_offsets=dst_offsets,
        buffers=buffers,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        #stream_comm=None,
        stream_comm = comm_stream

    )
    comm_stream.synchronize()  
    # Ensure all ranks have launched/completed their kernels before we start waiting locally.
    dist.barrier()

    # Device-side wait for token_sync (per expert) + acquire fence.
    iris_wait_token_sync_kernel[(e_local,)](
        buffers.token_sync,
        buffers.heap_bases,
        src_rank=rank,
        world_size=world_size,
        e_local=e_local,
        num_warps=1,
    )
    torch.cuda.synchronize()

    # run baseline
     # total_recv is known from counts_all (should match what step1 will produce)
    total_recv = int(counts_all[:, rank, :].sum().item())
    base_buffers = init_baseline_buffers(
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
        device=device,
        total_recv=total_recv,
        allocate_token_buf=True,
    )

    torch_out, _, _ = run_baseline_ref(
        rank=rank,
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        send_payload=send_payload,
        send_counts=send_counts,
        buffers=base_buffers,
        do_reorder=True,      # IMPORTANT: returns [E, world, CAP, H]
        profile=False,        # correctness run: don't care timings
        strict_capacity=True,
        barrier=True,
    )
    assert torch_out is not None
    torch.cuda.synchronize()

    # compare PCA
    pca_diff = (buffers.pca - expected_pca).abs().max().to(torch.float32)

    # compare token_buf (masked on valid region only) 
    max_diff, sum_triton, sum_torch = _masked_stats(buffers.token_buf, torch_out, expected_pca)

    # reduce to global stats
    pca_diff_global = pca_diff.clone()
    max_diff_global = max_diff.clone()
    sum_triton_global = sum_triton.clone()
    sum_torch_global = sum_torch.clone()

    dist.all_reduce(pca_diff_global, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_diff_global, op=dist.ReduceOp.MAX)
    dist.all_reduce(sum_triton_global, op=dist.ReduceOp.SUM)
    dist.all_reduce(sum_torch_global, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("=== Step1/2 Correctness Report ===")
        print(f"world={world_size} E_local={e_local} CAP={capacity} H={hidden_dim} batch={batch} seq={seq} topk={topk}")
        print(f"PCA max|diff| (global): {pca_diff_global.item():.6g}")
        print(f"TOKEN_BUF max|diff| over valid region (global): {max_diff_global.item():.6g}")
        print(f"TOKEN_BUF sum(valid) Triton (global): {sum_triton_global.item():.6g}")
        print(f"TOKEN_BUF sum(valid) Torch  (global): {sum_torch_global.item():.6g}")
        ok = (pca_diff_global.item() == 0.0) and (max_diff_global.item() == 0.0) and (
            abs(sum_triton_global.item() - sum_torch_global.item()) < 1e-3
        )
        print("PASS" if ok else "FAIL")

    dist.destroy_process_group()


if __name__ == "__main__":
    # Defaults for quick local testing (single node, WORLD_SIZE GPUs).
    # You can override via env vars.
    world_size = int(os.getenv("WORLD_SIZE", "2"))
    batch = int(os.getenv("BATCH", "4"))
    seq = int(os.getenv("SEQ", "128"))
    hidden_dim = int(os.getenv("HIDDEN", "64"))
    topk = int(os.getenv("TOPK", "2"))
    e_local = int(os.getenv("E_LOCAL", "4"))
    capacity = int(os.getenv("CAPACITY", "1024"))
    seed = int(os.getenv("SEED", "42"))

    # mp.spawn local init needs a rendezvous address.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.spawn(
        check_compare,
        args=(world_size, batch, seq, hidden_dim, topk, e_local, capacity, seed),
        nprocs=world_size,
        join=True,
    )
