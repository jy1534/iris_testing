import torch
import torch.distributed as dist

import os

import iris


from .layers.all_to_all import custom_a2a
from .layers.token_shuffle import shuffle
from .layers.expert import expert

from utils import alloc_counts_buffers, alloc_token_buffers


## Some simple testing utility functions.

def is_correct(one, two, threshold):
    if one.shape != two.shape:
        return abs(one.sum() - two.sum()) < threshold

    return torch.allclose(one, two, rtol=threshold)

def _require_dist():
    assert dist.is_initialized(), (
        "torch.distributed is not initialized. "
        "Run with torchrun/srun so process group is initialized before running unit tests."
    )
    assert dist.get_world_size() > 1, "Need WORLD_SIZE > 1 for custom_a2a unit test."
    assert torch.cuda.is_available(), "CUDA is required for this test."


def _get_shmem():
    if hasattr(iris, "shmem"):
        return iris.shmem
    if hasattr(iris, "symm_mem"):
        return iris.symm_mem
    raise RuntimeError("Cannot find shmem handle (expected iris.shmem or iris.symm_mem)")

def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0,high=100, (num_local_experts,))

    tokens = torch.randn(expert_token_cnt.sum(), token_hid_dim)

    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)

    return tokens, weights



def test_custom_a2a(e_local: int = 2, hidden_dim: int = 128, cap: int = 32, threshold: float = 1e-2) -> bool:
    """
       Routing pattern:
      Each src sends exactly `cap` rows to every (dst, local_expert).
      This makes expected placement deterministic and easy to check.
    """
    _require_dist()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    experts = e_local * world_size

    # deterministic routing metadata
    dest_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)
    dst_offsets = (torch.arange(world_size, device="cuda", dtype=torch.int32) * (e_local * cap)).contiguous()

    # build unique-valued tokens so misplacement is detectable ---
    total_rows = world_size * e_local * cap
    tokens = torch.empty((total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for dst in range(world_size):
        base = int(dst_offsets[dst].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + dst * 10_000 + e * 100 + t)
                tokens[row].fill_(val)

    # symmetric buffers
    shmem = _get_shmem()
    cb = alloc_counts_buffers(shmem, world_size=world_size, e_local=e_local)
    tb = alloc_token_buffers(
        shmem,
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=tokens.dtype,
    )

    cb.pca.zero_()
    cb.counts_ready.zero_()
    tb.token_buf.zero_()
    tb.token_sync.zero_()

    # the layerop
    out = custom_a2a(
        tokens,
        dest_counts,
        dst_offsets,
        cb.pca,
        tb.token_buf,
        cb.counts_ready,
        tb.token_sync,
        cb.heap_bases,
        experts,
        cap,
    )

    # wait via sync vars without barrier or sychornization
    while int(cb.counts_ready.item()) < world_size:
        pass
    while not bool(torch.all(tb.token_sync == world_size).item()):
        pass

    # gather inputs for expected mapping 
    gathered_in = [torch.empty_like(tokens) for _ in range(world_size)]
    dist.all_gather(gathered_in, tokens)

    # quick smoke: sums
    total_in_sum = torch.stack([x.sum() for x in gathered_in]).sum()
    total_out_sum = out.sum()
    assert is_correct(total_out_sum, total_in_sum, threshold), "SUM sanity check failed"

    #strong check: exact block placement for dst = rank
    dst_base = int(dst_offsets[rank].item())
    for src in range(world_size):
        src_tokens = gathered_in[src]
        for e in range(e_local):
            exp = src_tokens[dst_base + e * cap: dst_base + (e + 1) * cap, :]
            got = out[e, src, :, :]
            assert torch.equal(got, exp), f"block mismatch dst={rank} src={src} e={e}"

    return True

   

def test_gemm(total_expert_cnt, token_hid_dim, expert_hid_dim):
    
    tokens, weights = gen_gemm_input(total_expert_cnt, token_hid_dim, expert_hid_dim)

    custom_output = expert(tokens, weights,expert_token_cnt,total_expert_cnt)

    ## We use pytorch as ground truth. ##
    torch_out = []
    tokens_seen = 0
    for i in range(total_expert_cnt):
        torch_out.append(torch.einsum('sd,df->sf', tokens[tokens_seen:expert_token_cnt[i], :], weights[i]))
        tokens_seen += expert_token_cnt[i]

    return is_correct(custom_output, torch.stack(torch_out), 1e-2)


if __name__ == '__main__':

    ## Some sample inputs to test out correctness. ##
    test_gemm(2, 24, 48)
    test_gemm(5, 128, 128)
    
    test_custom_a2a(e_local=2, hidden_dim=128, cap=32)
    test_custom_a2a(e_local=4, hidden_dim=256, cap=16)
    

