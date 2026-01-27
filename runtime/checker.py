from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch


@dataclass(frozen=True)
class CheckConfig:
    atol: float = 0.0
    rtol: float = 0.0 # 4 float compare

    check_token_buf: bool = True
    check_pca: bool = True


def masked_stats(x: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """Compute simple stats over masked values."""
    v = x[mask]
    if v.numel() == 0:
        return {"sum": 0.0, "max_abs": 0.0}
    return {"sum": float(v.sum().item()), "max_abs": float(v.abs().max().item())}


def compare_pca(local_pca: torch.Tensor, expected_pca: torch.Tensor) -> Dict[str, Any]:
    """Checking counts exchange """
    diff = (local_pca.to(torch.int64) - expected_pca.to(torch.int64)).abs() # difference between counts from custom and ground truth after all_gather
    return {"pca_max_abs_diff": int(diff.max().item())}


def compare_token_buf_valid_region(
    token_buf_custom: torch.Tensor,
    token_buf_ref: torch.Tensor,
    expected_pca: torch.Tensor,
) -> Dict[str, Any]:
    """Compare token_buf within valid region defined by expected_pca[e,src] counts.

    Assumes token_buf shapes: [E, world, CAP, H].
    """
    E, W, CAP, H = token_buf_custom.shape
    assert token_buf_ref.shape == token_buf_custom.shape
    # Build a boolean mask [E, W, CAP] for valid rows
    # This is correctness-only; CPU mask is ok for clarity.
    counts = expected_pca.detach().cpu().to(torch.int64)  # [E,W]
    mask = torch.zeros((E, W, CAP), dtype=torch.bool)
    for e in range(E):
        for s in range(W):
            n = int(counts[e, s].item())
            n = max(0, min(CAP, n))
            if n:
                mask[e, s, :n] = True

    # Compare sums and max diff over valid region
    x = token_buf_custom.detach().cpu().to(torch.float32)
    y = token_buf_ref.detach().cpu().to(torch.float32)
    diff = (x - y).abs()

    # Expand mask to [E,W,CAP,H]
    mask4 = mask.unsqueeze(-1).expand(-1, -1, -1, H)
    vdiff = diff[mask4]
    return {
        "token_buf_valid_max_abs_diff": float(vdiff.max().item()) if vdiff.numel() else 0.0,
        "token_buf_valid_sum_custom": float(x[mask4].sum().item()) if vdiff.numel() else 0.0,
        "token_buf_valid_sum_ref": float(y[mask4].sum().item()) if vdiff.numel() else 0.0,
    }
