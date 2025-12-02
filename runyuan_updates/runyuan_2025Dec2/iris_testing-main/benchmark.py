#import torch.multiprocessing as mp
#import torch
#import torch.distributed as dist
#import torch.nn.functional as F

#from utils import gen_tensor
#from alltoall_persistent_gemm import run as CustomA2A 
#from baseline import run as TorchA2A

#new imports
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import gen_tensor, gen_expert_weights
from alltoall_persistent_gemm import run as CustomA2A
from baseline import run as TorchA2A

import triton
import triton.language as tl

import sys
import os
import time

import iris

#def callee(
#    rank: int, batch: int, seq: int, hidden_dim: int, num_experts: int,
#    world_size: int, topk: int, opt: bool 
#    ):
def callee(
    rank: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    topk: int,
    opt: bool,
    expert_weights_cpu: torch.Tensor,
):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
    opt: control whether to trigger Custom A2A or baseline, respectively.
    """
#    device_id = rank % torch.cuda.device_count()
#    dist.init_process_group(
#        backend="nccl",
#        rank=rank,
#        world_size=world_size,
#        init_method="tcp://127.0.0.1:29500",
#        device_id=torch.device(f"cuda:{device_id}")
#    )

#    heap_size = (2**30)*100 ## 1 GiB symmetric heap.
#    shmem = iris.iris(heap_size)

#    tokens, chunk_size = gen_tensor(
#        batch, seq, hidden_dim, 
#        world_size, num_experts, 
#        rank, topk)
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)# set cpu used in this process

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
    )

    heap_size = (2**30) * 100  # 1 GiB symmetric heap.
    shmem = iris.iris(heap_size)

    tokens, chunk_size = gen_tensor(
        batch, seq, hidden_dim,
        world_size, num_experts,
        rank, topk,
    )

    # cpu weigh in main to cpu weigh in rank
    expert_weights = expert_weights_cpu.to(device=tokens.device)


    if rank == 0:
        print(f'[rank: {rank}], chunk_size: {chunk_size}')
    ## Warmup. 
#    for _ in range(5):
#        if opt:
#            ## Currently meta is an empty list. Will be required for uneven all-to-all later. ##
#            CustomA2A(
#                rank, tokens, chunk_size, batch, 
#                seq, hidden_dim, num_experts, 
#                world_size, shmem, False
#                )
#        else:
            ## Currently meta is an empty list. Will be required for uneven all-to-all later. ##
#            TorchA2A(
#                rank, tokens, chunk_size, batch,
#                seq, hidden_dim, num_experts,
#                world_size, False, shmem
#                )
#        shmem.barrier()
    for _ in range(5):
        if opt:
            CustomA2A(
                rank, tokens, expert_weights, chunk_size,
                batch, seq, hidden_dim, num_experts,
                world_size, shmem, False,
            )
        else:
            TorchA2A(
                rank, tokens, expert_weights, chunk_size,
                batch, seq, hidden_dim, num_experts,
                world_size, False, shmem,
            )
        shmem.barrier()

    ##timing  
#    for _ in range(10):
#        if opt:
#            CustomA2A(
#                rank, tokens, chunk_size, batch, 
#                seq, hidden_dim, num_experts, 
#                world_size, shmem, False
#                )
#        else:
#            TorchA2A(
#                rank, tokens, chunk_size, batch,
#                seq, hidden_dim, num_experts,
#                world_size, False, shmem
#                )

#    shmem.barrier()
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        if opt:
            CustomA2A(
                rank, tokens, expert_weights, chunk_size,
                batch, seq, hidden_dim, num_experts,
                world_size, shmem, False,
            )
        else:
            TorchA2A(
                rank, tokens, expert_weights, chunk_size,
                batch, seq, hidden_dim, num_experts,
                world_size, False, shmem,
            )
        shmem.barrier()

    torch.cuda.synchronize()
    end_time = time.time()
    print(f'[rank: {rank}] time taken: {(end_time - start_time):.5f}')
    dist.destroy_process_group()

if __name__ == "__main__":
    ## Input parameters. ##
    world_size, batch, seq, hidden_dim, topk = 8, 4, 1024, 768, 2
    num_experts = world_size * 32  ## Multiple experts per device, evenly distributed. ##
    run_custom_a2a: bool =  True
#    ## A custom test case for convenience. ##
#    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size, topk, run_custom_a2a), nprocs=world_size, join=True)
    ## Generate expert weights once in main (CPU), then broadcast as arg. ##
    expert_dim = hidden_dim  # simplest case: square FFN, can be changed later
    device_cpu = torch.device("cpu")
    dtype = torch.bfloat16
    expert_weights = gen_expert_weights(
        num_experts,
        hidden_dim,
        expert_dim,
        device_cpu,
        dtype,
    )

    ## A custom test case for convenience. 
    mp.spawn(
        callee,
        args=(batch, seq, hidden_dim, num_experts, world_size, topk, run_custom_a2a, expert_weights),
        nprocs=world_size,
        join=True,
    )