from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TimeConfig:
    warmup: int = 5
    iters: int = 20
    # default close incase it may pollute the result
    barrier_before: bool = False
    barrier_after: bool = False
    # Fairness: do all_reduce(MAX) per iter to get global iter time
    reduce_max_across_ranks: bool = True


def _quantile_95(t: torch.Tensor) -> float:
    # t is 1D CPU float64 tensor
    if t.numel() == 1:
        return float(t.item())
    # Prefer quantile if available
    if hasattr(torch, "quantile"):
        return float(torch.quantile(t, 0.95).item())
    # Fallback: kthvalue with ceil (1-indexed)
    k = int(math.ceil(0.95 * t.numel()))
    k = max(1, min(k, t.numel()))
    return float(t.kthvalue(k).values.item())


def time_cuda_ms(
    fn: Callable[[], None],
    *,
    cfg: TimeConfig = TimeConfig(),
    barrier_fn: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Time a callable using CUDA events.

    Fairness (Scheme A):
      - Each iter measures local CUDA-event time
      - Then all_reduce(MAX) to get global iter time (slowest rank)
      - Stats are computed on the global iter times

      - `fn()` MUST include completion fence semantics (e.g., token_sync wait or shmem.barrier)
    """
    # Warmup (not timed)
    for _ in range(cfg.warmup):
        fn()

    # Optional alignment (generally disable for perf per Ahan)
    if barrier_fn is not None and cfg.barrier_before:
        barrier_fn()
    torch.cuda.synchronize()

    times: List[float] = []

    do_reduce = (
        cfg.reduce_max_across_ranks
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )

    for _ in range(cfg.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()
        end.synchronize()

        local_ms = float(start.elapsed_time(end))

        if do_reduce:
            # All-reduce MAX across ranks for this iter's global time
            t_ms = torch.tensor([local_ms], device="cuda", dtype=torch.float32)
            dist.all_reduce(t_ms, op=dist.ReduceOp.MAX)
            global_ms = float(t_ms.item())
            times.append(global_ms)
        else:
            times.append(local_ms)

    if barrier_fn is not None and cfg.barrier_after:
        barrier_fn()
    torch.cuda.synchronize()

    t = torch.tensor(times, dtype=torch.float64)  # CPU stats
    return {
        "mean_ms": float(t.mean().item()),
        "p50_ms": float(t.median().item()),
        "p95_ms": _quantile_95(t),
        "min_ms": float(t.min().item()),
        "max_ms": float(t.max().item()),
        "iters": int(cfg.iters),
        "warmup": int(cfg.warmup),
        "reduce_max_across_ranks": bool(cfg.reduce_max_across_ranks),
     }