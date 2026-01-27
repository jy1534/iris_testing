from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class SyncConfig:
    timeout_s: float = 30.0


def reset_shmem_buffers(*, buffers, clear_token_buf: bool = False) -> None:
    """SSOT reset: zero sync vars (and optionally token_buf) before each iteration."""
    buffers.pca.zero_()
    buffers.counts_ready.zero_()
    buffers.token_sync.zero_()
    if clear_token_buf:
        buffers.token_buf.zero_()


def expected_token_tiles(pca_e_by_src: torch.Tensor, block_m: int) -> torch.Tensor:
    """Compute expected tile increments per expert: sum_src ceil(pca[e,src]/block_m)."""
    assert pca_e_by_src.dtype == torch.int32
    # ceil_div for int32 on GPU; careful with negatives (should not exist)
    bm = int(block_m)
    return (pca_e_by_src.add(bm - 1).div(bm, rounding_mode="floor")).sum(dim=1).to(torch.int32)


def wait_counts_ready(counts_ready: torch.Tensor, world_size: int, cfg: SyncConfig = SyncConfig()) -> None:
    """Spin until counts_ready == world_size (Stage-1 completion)."""
    t0 = time.time()
    while True:
        # .item() introduces a sync; ok for tests/correctness
        v = int(counts_ready.item())
        if v >= world_size:
            return
        if time.time() - t0 > cfg.timeout_s:
            raise TimeoutError(f"counts_ready timeout: got {v}, want {world_size}")
        time.sleep(0.0001)


def wait_tokens_ready(token_sync: torch.Tensor, expected_tiles: torch.Tensor, cfg: SyncConfig = SyncConfig()) -> None:
    """Spin until token_sync[e] >= expected_tiles[e] for all experts."""
    t0 = time.time()
    # Move expected to CPU once
    exp = expected_tiles.detach().cpu().to(torch.int64)
    while True:
        cur = token_sync.detach().cpu().to(torch.int64)
        if torch.all(cur >= exp):
            return
        if time.time() - t0 > cfg.timeout_s:
            bad = (cur < exp).nonzero(as_tuple=False).flatten().tolist()
            raise TimeoutError(f"token_sync timeout on experts={bad}; cur={cur.tolist()} exp={exp.tolist()}")
        time.sleep(0.0001)
