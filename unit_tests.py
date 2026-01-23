import torch
import torch.distributed as dist

import os

import iris


from .layers.all_to_all import custom_a2a
from .layers.token_shuffle import shuffle
from .layers.expert import expert

from kernels import counts_exchange_kernel, tokens_exchange_kernel
from utils import (
    alloc_counts_buffers,
    alloc_token_buffers,
    build_expert_offsets,
)


## Some simple testing utility functions.

def is_correct(one, two, threshold):
    if one.shape != two.shape:
        return abs(one.sum() - two.sum()) < threshold

    return torch.allclose(one, two, rtol=threshold)

def _env_int(*names: str, default: int) -> int:
    for n in names:
        v = os.getenv(n, None)
        if v is not None:
            return int(v)
    return int(default)

def _init_dist():
    if dist.is_initialized():
        return
    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)
    if world_size <= 1:
        return
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

def _get_shmem():
    # keep it simple: adapt if your repo exposes differently
    if hasattr(iris, "shmem"):
        return iris.shmem
    if hasattr(iris, "symm_mem"):
        return iris.symm_mem
    raise RuntimeError("Cannot find shmem handle (expected iris.shmem or iris.symm_mem)")



def gen_tensor(shape, cuda=True):
    return torch.randn(*shape, dtype=torch.bfloat16).to("cuda" if cuda else "cpu")


def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0,high=100, (num_local_experts,))

    tokens = torch.randn(expert_token_cnt.sum(), token_hid_dim)

    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)

    return tokens, weights


def test_counts_exchange_kernel(e_local: int = 4):
    assert dist.is_initialized()
    assert torch.cuda.is_available()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shmem = _get_shmem()

    # unique counts per src rank
    dest_counts = (
        torch.arange(world_size * e_local, device="cuda", dtype=torch.int32).reshape(world_size, e_local)
        + rank * 10000
    )

    cb = alloc_counts_buffers(shmem, world_size=world_size, e_local=e_local)
    cb.counts_ready.zero_()
    torch.cuda.synchronize()
    dist.barrier()

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

    # expected pca on dst=rank: column src is src.dest_counts[dst, :]
    gathered = [torch.empty_like(dest_counts) for _ in range(world_size)]
    dist.all_gather(gathered, dest_counts)

    expected = torch.empty((e_local, world_size), device="cuda", dtype=torch.int32)
    for src in range(world_size):
        expected[:, src] = gathered[src][rank, :]

    assert torch.equal(cb.pca, expected), f"pca mismatch on rank {rank}"
    assert int(cb.counts_ready.item()) == world_size, f"counts_ready mismatch on rank {rank}"

def test_tokens_exchange_kernel(e_local: int = 2, cap: int = 32, hidden_dim: int = 128):
    assert dist.is_initialized()
    assert torch.cuda.is_available()

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    shmem = _get_shmem()

    # uniform counts => easiest expected
    send_counts = torch.full((world_size, e_local), cap, device="cuda", dtype=torch.int32)
    expert_offs = build_expert_offsets(send_counts)  # [world, e_local], expert_offs[dst,e] = e*cap

    block = e_local * cap
    dst_offsets = (torch.arange(world_size, device="cuda", dtype=torch.int32) * block).contiguous()

    # send_payload packed by [dst][expert][t]
    total_rows = world_size * block
    send_payload = torch.empty((total_rows, hidden_dim), device="cuda", dtype=torch.bfloat16)
    for d in range(world_size):
        base = int(dst_offsets[d].item())
        for e in range(e_local):
            for t in range(cap):
                row = base + e * cap + t
                val = float(rank * 1_000_000 + d * 10_000 + e * 100 + t)
                send_payload[row, :].fill_(val)

    # allocate token buffers (heap_bases is obtained once)
    heap_bases = shmem.get_heap_bases()
    tb = alloc_token_buffers(
        shmem,
        world_size=world_size,
        e_local=e_local,
        capacity=cap,
        hidden_dim=hidden_dim,
        token_dtype=send_payload.dtype,
    )
    tb.token_buf.zero_()
    tb.token_sync.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    tokens_exchange_kernel[(world_size, e_local)](
        send_payload,
        send_counts,
        dst_offsets,
        expert_offs,
        tb.token_buf,
        tb.token_sync,
        heap_bases,
        src_rank=rank,
        world_size=world_size,
        e_local=e_local,
        CAP=cap,
        hidden_dim=hidden_dim,
        BLOCK_M=32,
        BLOCK_K=128,
        num_warps=8,
    )

    torch.cuda.synchronize()
    dist.barrier()

    # gather payload to build expected
    gathered_payload = [torch.empty_like(send_payload) for _ in range(world_size)]
    dist.all_gather(gathered_payload, send_payload)

    # on dst=rank, token_buf[e, src, :, :] equals src payload slice for dst block + expert offset
    for src in range(world_size):
        for e in range(e_local):
            base = int(dst_offsets[rank].item()) + e * cap
            expected = gathered_payload[src][base:base+cap, :]
            got = tb.token_buf[e, src, :, :]
            assert torch.equal(got, expected), f"token_buf mismatch dst={rank} src={src} expert={e}"

    assert torch.equal(tb.token_sync, torch.full((e_local,), world_size, device="cuda", dtype=torch.int32))




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
    
    _init_dist()
    if dist.is_initialized() and dist.get_world_size() > 1:
        test_counts_exchange_kernel(e_local=4)
        test_tokens_exchange_kernel(e_local=2, cap=32, hidden_dim=128)
        dist.barrier()
    

