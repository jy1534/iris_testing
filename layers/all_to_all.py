import torch
import torch.distributed as dist

from .kernels import counts_exchange_kernel, tokens_exchange_kernel
from .utils import (
    build_expert_offsets,
    alloc_counts_buffers,
    alloc_token_buffers,
    compute_capacity_from_pca,
)

def custom_a2a(
    shmem,
    tokens: torch.Tensor,        # [sum_send, H]
    dest_counts: torch.Tensor,   # [world, E_local] int32
    dst_offsets: torch.Tensor,   # [world] int32
    experts: int,                # total experts
):
    """
    Two-stage SHMEM All-to-All:
      Step-1: counts exchange -> writes pca on dst
      CAP: max(pca) global-reduced
      Step-2: token exchange -> writes token_buf on dst
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert experts % world_size == 0, "experts must be divisible by world_size"
    e_local = experts // world_size
    hidden_dim = tokens.shape[-1]

    # Stage-1 alloc
    cb = alloc_counts_buffers(shmem, world_size=world_size, e_local=e_local)
    dist.barrier()  # keep symmetric heap order aligned

    # Step-1 kernel
    counts_exchange_kernel[(world_size,)](
        dest_counts,
        cb.pca,
        cb.counts_ready,
        cb.heap_bases,
        src_rank=rank,
        world_size=world_size,
        e_local=e_local,
        BLOCK_E=128,
        num_warps=4,
    )

    torch.cuda.synchronize()
    dist.barrier()

    # CAP from pca
    cap = compute_capacity_from_pca(cb.pca)

    # Stage-2 alloc (post Step-1)
    tb = alloc_token_buffers(
        shmem,
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
    )
    dist.barrier()

    # Step-2 kernel
    expert_offs = build_expert_offsets(dest_counts)
    tokens_exchange_kernel[(world_size, e_local)](
        tokens,
        dest_counts,
        dst_offsets,
        expert_offs,
        tb.token_buf,
        tb.token_sync,
        cb.heap_bases,
        src_rank=rank,
        world_size=world_size,
        e_local=e_local,
        CAP=cap,
        hidden_dim=hidden_dim,
        BLOCK_M=32,
        BLOCK_K=128,
        num_warps=8,
    )

    return tb.token_buf, tb.token_sync, cb.pca, cap
