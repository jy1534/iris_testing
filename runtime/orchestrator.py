from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.distributed as dist
import iris

from utils import (
    set_seed,
    gen_local_tokens,
    gen_router,
    route_and_pack_padding_free,
    alloc_shmem_buffers,
)
from baseline import init_baseline_buffers

from runtime.sync import reset_shmem_buffers
from runtime.timer import time_cuda_ms, TimeConfig
from runtime.executor import (
    run_custom_launch,
    run_custom_fence,
    run_baseline,
    ExecConfig,
)
from ops.a2a_core import A2AConfig, KernelConfig
from runtime.checker import compare_pca, compare_token_buf_valid_region


@dataclass(frozen=True)
class CaseConfig:
    batch: int
    seq: int
    hidden_dim: int
    topk: int
    e_local: int
    capacity: int
    seed: int = 42
    kernel: KernelConfig = KernelConfig()


def init_iris_shmem() -> Any:
    # Keep behavior aligned with existing benchmark's iris init
    return iris.init()


def build_dst_offsets(send_counts: torch.Tensor) -> torch.Tensor:
    """
    send_counts: [world, E_local] int32
    returns dst_offsets: [world] int32
    where dst_offsets[dst] is row offset of dst-block within send_payload.
    """
    assert send_counts.dtype == torch.int32 and send_counts.is_cuda
    dst_sizes = send_counts.sum(dim=1).to(torch.int32)  # [world]
    dst_offsets = torch.cumsum(dst_sizes, dim=0) - dst_sizes
    return dst_offsets.contiguous()


def run_case(
    *,
    rank: int,
    world_size: int,
    case: CaseConfig,
    mode: str = "both",  # "perf" | "correctness" | "both"
    warmup: int = 5,
    iters: int = 20,
    clear_token_buf: bool = False,
) -> Dict[str, Any]:
    # distributed init
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    set_seed(case.seed, rank)
    experts_total = world_size * case.e_local

    # inputs
    tokens = gen_local_tokens(
        case.batch, case.seq, case.hidden_dim,
        torch.bfloat16, device, case.seed, rank
    )
    router = gen_router(case.hidden_dim, experts_total, torch.bfloat16, device, case.seed)

    send_payload, send_counts, _, _ = route_and_pack_padding_free(
        tokens, router, case.topk, world_size, experts_total
    )
    send_counts = send_counts.to(torch.int32).contiguous()
    dst_offsets = build_dst_offsets(send_counts)

    # correctness ground truth: expected_pca[e, src] for this dst=rank 
    all_send_counts = [torch.zeros_like(send_counts) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    counts_all = torch.stack(all_send_counts, dim=0).contiguous()  # [src,dst,E]
    expected_pca = counts_all[:, rank, :].transpose(0, 1).contiguous()  # [E,world]

    # shmem buffers for custom 
    shmem = init_iris_shmem()
    buffers = alloc_shmem_buffers(
        shmem=shmem,
        world_size=world_size,
        e_local=case.e_local,
        capacity=case.capacity,
        hidden_dim=case.hidden_dim,
        token_dtype=torch.bfloat16,
    )

    #  baseline buffers
    total_recv = int(counts_all[:, rank, :].sum().item())
    base_buffers = init_baseline_buffers(
        world_size=world_size,
        e_local=case.e_local,
        capacity=case.capacity,
        hidden_dim=case.hidden_dim,
        token_dtype=send_payload.dtype,
        device=device,
        total_recv=total_recv,
        allocate_token_buf=(mode in ("correctness", "both")),
    )

    a2a_cfg = A2AConfig(experts_total=experts_total, capacity=case.capacity, kernel=case.kernel)

    result: Dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "case": {
            "batch": case.batch,
            "seq": case.seq,
            "hidden_dim": case.hidden_dim,
            "topk": case.topk,
            "e_local": case.e_local,
            "capacity": case.capacity,
        },
    }

   
    # PERF (fair semantics)
    
    if mode in ("perf", "both"):

        # Custom perf: launch + local completion fence via token_sync
        def custom_iter():
            reset_shmem_buffers(buffers=buffers, clear_token_buf=clear_token_buf)
            run_custom_launch(
                buffers=buffers,
                send_payload=send_payload,
                send_counts=send_counts,
                dst_offsets=dst_offsets,
                a2a_cfg=a2a_cfg,
                stream=None,
            )
            run_custom_fence(
                shmem=shmem,
                buffers=buffers,
                a2a_cfg=a2a_cfg,
                fence="shmem_barrier",  # perf: completion fence (stable); Scheme A fairness via all_reduce(MAX)
            )

        # Timer Scheme A already does all_reduce(MAX) per-iter if dist is initialized.
        result["custom_perf"] = time_cuda_ms(
            custom_iter,
            cfg=TimeConfig(
                warmup=warmup,
                iters=iters,
                barrier_before=False,
                barrier_after=False,
                reduce_max_across_ranks=True,
            ),
            barrier_fn=None,
        )

        # Baseline perf: no reorder; no internal barrier; local completion is "collective returns"
        def baseline_iter():
            run_baseline(
                rank=rank,
                world_size=world_size,
                e_local=case.e_local,
                capacity=case.capacity,
                hidden_dim=case.hidden_dim,
                send_payload=send_payload,
                send_counts=send_counts,
                buffers=base_buffers,
                exec_cfg=ExecConfig(do_reorder=False, strict_capacity=False),
            )

        result["baseline_perf"] = time_cuda_ms(
            baseline_iter,
            cfg=TimeConfig(
                warmup=warmup,
                iters=iters,
                barrier_before=False,
                barrier_after=False,
                reduce_max_across_ranks=True,
            ),
            barrier_fn=None,
        )

                         
    # CORRECTNESS (robust fence)
      
    if mode in ("correctness", "both"):
        reset_shmem_buffers(buffers=buffers, clear_token_buf=clear_token_buf)

        run_custom_launch(
            buffers=buffers,
            send_payload=send_payload,
            send_counts=send_counts,
            dst_offsets=dst_offsets,
            a2a_cfg=a2a_cfg,
            stream=None,
        )
        # Correctness: simplest robust global fence
        run_custom_fence(
            shmem=shmem,
            buffers=buffers,
            a2a_cfg=a2a_cfg,
            fence="shmem_barrier",
        )

        # Baseline with reorder for token_buf comparison
        run_baseline(
            rank=rank,
            world_size=world_size,
            e_local=case.e_local,
            capacity=case.capacity,
            hidden_dim=case.hidden_dim,
            send_payload=send_payload,
            send_counts=send_counts,
            buffers=base_buffers,
            exec_cfg=ExecConfig(do_reorder=True, strict_capacity=True),
        )

        # Checks
        result["check_pca"] = compare_pca(
            buffers.pca,
            expected_pca.to(device=buffers.pca.device, dtype=buffers.pca.dtype),
        )
        result["check_token_buf"] = compare_token_buf_valid_region(
            buffers.token_buf,
            base_buffers.token_buf,
            expected_pca,
        )

    # Teardown safety: keep one barrier so ranks don't exit unevenly
    dist.barrier()
    dist.destroy_process_group()
    return result
