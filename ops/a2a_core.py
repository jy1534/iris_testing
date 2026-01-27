from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import triton

from kernels import counts_exchange_kernel, tokens_exchange_kernel
from utils import build_expert_offsets, _assert_cuda_int32


@dataclass(frozen=True)
class KernelConfig:
    # Stage-1
    block_e: int = 128
    stage1_num_warps: int = 4

    # Stage-2
    block_m: int = 32
    block_k: int = 128
    stage2_num_warps: int = 8


@dataclass(frozen=True)
class A2AConfig:
    experts_total: int
    capacity: int
    kernel: KernelConfig = KernelConfig()


def a2a_forward(
    *,
    tokens: torch.Tensor,             # [sum_send, H]
    dest_counts: torch.Tensor,        # [world, E_local] int32
    dst_offsets: torch.Tensor,        # [world] int32
    local_pca: torch.Tensor,          # [E_local, world] int32 (symmetric)
    token_buf: torch.Tensor,          # [E_local, world, CAP, H] (symmetric)
    counts_ready: torch.Tensor,       # [1] int32 (symmetric)
    token_sync: torch.Tensor,         # [E_local] int32 (symmetric)
    heap_bases: torch.Tensor,         # [world] (implementation-dependent)
    cfg: A2AConfig,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Single source of truth for the custom shmem-based 2-stage A2A.

    Notes:
      - This function performs kernel launches only.
      - No dist.barrier(), no torch.cuda.synchronize().
      - Completion semantics are handled by the caller via shmem.barrier() or spin-wait.
    """
    _assert_cuda_int32(dest_counts, "dest_counts")
    _assert_cuda_int32(dst_offsets, "dst_offsets")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    assert cfg.experts_total % world_size == 0, "experts_total must be divisible by world_size"
    e_local = cfg.experts_total // world_size
    hidden_dim = tokens.shape[-1]

    # Stage-1: counts exchange
    if stream is None:
        counts_exchange_kernel[(world_size,)](
            dest_counts,
            local_pca,
            counts_ready,
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            BLOCK_E=cfg.kernel.block_e,
            num_warps=cfg.kernel.stage1_num_warps,
        )
    else:
        with torch.cuda.stream(stream):
            counts_exchange_kernel[(world_size,)](
                dest_counts,
                local_pca,
                counts_ready,
                heap_bases,
                src_rank=rank,
                world_size=world_size,
                e_local=e_local,
                BLOCK_E=cfg.kernel.block_e,
                num_warps=cfg.kernel.stage1_num_warps,
            )

    # Stage-2: token exchange
    expert_offs = build_expert_offsets(dest_counts)  # [world, E_local] int32 prefix offsets within dst segment
    grid_m = triton.cdiv(cfg.capacity, cfg.kernel.block_m)

    if stream is None:
        tokens_exchange_kernel[(world_size, e_local, grid_m)](
            tokens,
            dest_counts,
            dst_offsets,
            expert_offs,
            token_buf,
            token_sync,
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=e_local,
            CAP=cfg.capacity,
            hidden_dim=hidden_dim,
            BLOCK_M=cfg.kernel.block_m,
            BLOCK_K=cfg.kernel.block_k,
            num_warps=cfg.kernel.stage2_num_warps,
        )
    else:
        with torch.cuda.stream(stream):
            tokens_exchange_kernel[(world_size, e_local, grid_m)](
                tokens,
                dest_counts,
                dst_offsets,
                expert_offs,
                token_buf,
                token_sync,
                heap_bases,
                src_rank=rank,
                world_size=world_size,
                e_local=e_local,
                CAP=cfg.capacity,
                hidden_dim=hidden_dim,
                BLOCK_M=cfg.kernel.block_m,
                BLOCK_K=cfg.kernel.block_k,
                num_warps=cfg.kernel.stage2_num_warps,
            )

    return token_buf
