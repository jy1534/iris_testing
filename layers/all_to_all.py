from __future__ import annotations

import torch
import torch.distributed as dist

from ops.a2a_core import a2a_forward, A2AConfig, KernelConfig


class AllToAllOp(torch.autograd.Function):
    """Thin façade to keep the 'layers.all_to_all' workflow stable.

    Forward delegates to ops.a2a_core.a2a_forward (SSOT).
    Backward intentionally not implemented.
    """

    @staticmethod
    def forward(
        ctx,
        tokens,
        dest_counts,
        dst_offsets,
        local_pca,
        token_buf,
        counts_ready,
        token_sync,
        heap_bases,
        experts_total: int,
        capacity: int,
        # Optional kernel knobs (keep default identical to current behavior)
        block_e: int = 128,
        block_m: int = 32,
        block_k: int = 128,
        stage1_num_warps: int = 4,
        stage2_num_warps: int = 8,
    ):
        cfg = A2AConfig(
            experts_total=experts_total,
            capacity=capacity,
            kernel=KernelConfig(
                block_e=block_e,
                stage1_num_warps=stage1_num_warps,
                block_m=block_m,
                block_k=block_k,
                stage2_num_warps=stage2_num_warps,
            ),
        )
        return a2a_forward(
            tokens=tokens,
            dest_counts=dest_counts,
            dst_offsets=dst_offsets,
            local_pca=local_pca,
            token_buf=token_buf,
            counts_ready=counts_ready,
            token_sync=token_sync,
            heap_bases=heap_bases,
            cfg=cfg,
            stream=None,
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass not implemented.")


custom_a2a = AllToAllOp.apply
