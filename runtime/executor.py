from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.distributed as dist

from ops.a2a_core import a2a_forward, A2AConfig
from baseline import run_baseline_ref

from runtime.sync import (
    SyncConfig,
    wait_counts_ready,
    expected_token_tiles,
    wait_tokens_ready,
)


@dataclass(frozen=True)
class ExecConfig:
    do_reorder: bool = False
    strict_capacity: bool = False


# Custom path: launch phase

def run_custom_launch(
    *,
    buffers,
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    dst_offsets: torch.Tensor,
    a2a_cfg: A2AConfig,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Launch Stage-1/2 kernels only.
    No barrier, no spin-wait, no synchronize.
    """
    out = a2a_forward(
        tokens=send_payload,
        dest_counts=send_counts,
        dst_offsets=dst_offsets,
        local_pca=buffers.pca,
        token_buf=buffers.token_buf,
        counts_ready=buffers.counts_ready,
        token_sync=buffers.token_sync,
        heap_bases=buffers.heap_bases,
        cfg=a2a_cfg,
        stream=stream,
    )
    return out


# Custom path: fence phase

def run_custom_fence(
    *,
    shmem,
    buffers,
    a2a_cfg: A2AConfig,
    fence: Literal["shmem_barrier", "token_sync"] = "shmem_barrier",
    sync_cfg: SyncConfig = SyncConfig(),
) -> None:
    """
    Completion fence for the custom path.

    - shmem_barrier: global fence (simple, reliable; can token-balance across ranks)
    - token_sync: local completion based on sync vars (preferred for perf fairness)
    """
    if fence == "shmem_barrier":
        shmem.barrier()
        return

    if fence == "token_sync":
        world_size = dist.get_world_size()

        # Ensure Stage-1 visible so pca is stable
        wait_counts_ready(buffers.counts_ready, world_size, cfg=sync_cfg)

        # Compute expected tiles per expert from pca[e, src]
        exp_tiles = expected_token_tiles(buffers.pca, block_m=a2a_cfg.kernel.block_m)

        # Wait until each expert's token tiles arrived
        wait_tokens_ready(buffers.token_sync, exp_tiles, cfg=sync_cfg)
        return

    raise ValueError(f"Unknown fence={fence}")



# Baseline path (still single call; fence handled outside if needed)

def run_baseline(
    *,
    rank: int,
    world_size: int,
    e_local: int,
    capacity: int,
    hidden_dim: int,
    send_payload: torch.Tensor,
    send_counts: torch.Tensor,
    buffers,
    exec_cfg: ExecConfig = ExecConfig(),
) -> None:
    """
    Baseline should not do dist.barrier() inside (fairness).
    Any extra fence/timing policy is applied by orchestrator/timer.
    """
    _ = run_baseline_ref(
        rank=rank,
        world_size=world_size,
        e_local=e_local,
        capacity=capacity,
        hidden_dim=hidden_dim,
        send_payload=send_payload,
        send_counts=send_counts,
        buffers=buffers,
        do_reorder=exec_cfg.do_reorder,
        profile=False,
        strict_capacity=exec_cfg.strict_capacity,
        barrier=False,  # do NOT sync inside baseline
    )
