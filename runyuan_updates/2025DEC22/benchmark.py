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
import time # already have

import iris
import os




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
    #device_id = rank % torch.cuda.device_count()
    #device = torch.device(f"cuda:{device_id}")
    #torch.cuda.set_device(device_id)  # set GPU used in this process

    #dist.init_process_group(
        #backend="nccl",
        #rank=rank,
        #world_size=world_size,
        #init_method="tcp://127.0.0.1:29500",
    #)

   # Safer size
    #heap_size = (2**30) * 10 # 10GB heap #100 before  # 1 GiB symmetric heap.
    
    #shmem = iris.iris(heap_size)

    #tokens, chunk_size = gen_tensor(
        #batch, seq, hidden_dim,
        #world_size, num_experts,
        #rank, topk,
    #)
    """
    in case there are always cache write error？？？  mod = importlib.util.module_from_spec(spec)
    File "<frozen importlib._bootstrap>", line 565, in module_from_spec
    File "<frozen importlib._bootstrap_external>", line 1174, in create_module
    File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
    ImportError: /home1/runyuan15/.triton/cache/3V3PL3ZJAOWBKOFOHDAEQ2QTSMGTAS3HNLC4QIPUIOTKYINHZMHQ/hip_utils.cpython-39-x86_64-linux-gnu.so: cannot open shared object file: No such file or directory
"""
    
    cache_dir = f"/tmp/{os.environ.get('USER', 'user')}_triton_rank{rank}"
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device_id)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
    )

    # First generate tokens and per-(src,dst) chunk_size (transmit_size)
    tokens, chunk_size = gen_tensor(
        batch, seq, hidden_dim,
        world_size, num_experts,
        rank, topk,
    )

    # Dynamically calculate Iris heap_size
    # In both CustomA2A and baseline, chunk_size is used as transmit_size.
    transmit_size = chunk_size
    hidden_dim_local = tokens.shape[-1]
    bytes_per_element = torch.finfo(tokens.dtype).bits // 8

    # routed_token_buffer size ~ transmit_size * world_size * hidden_dim * sizeof(dtype)
    MAX_GRID_Y = 60000
    max_chunk = min(transmit_size, MAX_GRID_Y)

    bytes_per_element = torch.finfo(tokens.dtype).bits // 8
    buffer_bytes = max_chunk * world_size * hidden_dim_local * bytes_per_element

    HEAP_FACTOR = 2.0
    MIN_HEAP_BYTES = 1 * (2**30)

    heap_size = int(buffer_bytes * HEAP_FACTOR)
    heap_size = max(heap_size, MIN_HEAP_BYTES)

    if rank == 0:
        print(
            f"[rank {rank}] Iris heap_size = {heap_size / (2**30):.2f} GiB "
            f"(buffer ~ {buffer_bytes / (2**30):.2f} GiB)"
        )

    # init Iris
    shmem = iris.iris(heap_size)

    # move expert weights to this rank's device
    expert_weights = expert_weights_cpu.to(device=tokens.device)

    if rank == 0:
        print(f"[rank: {rank}], chunk_size: {chunk_size}")

    # GEMM buffer switch 
    # True: GEMM dreectly on IRIS symmetric heap
    # False: A2A with IRIS buffer，clone GEMM to ord CUDA first
    use_iris_buffer_for_gemm = False

    # M1: do only one correctness check
    debug_check = False

    if debug_check:
        # all rank run baseline + custom，print only at rank 0 
        out_baseline = TorchA2A(
            rank,
            tokens,
            expert_weights,
            chunk_size,
            batch,
            seq,
            hidden_dim,
            num_experts,
            world_size,
            False,   # general_a2a
            shmem,
        )
        out_custom = CustomA2A(
            rank,
            tokens,
            expert_weights,
            chunk_size,
            batch,
            seq,
            hidden_dim,
            num_experts,
            world_size,
            shmem,
            False,   # general_a2a
            # default profile: False
            use_iris_buffer_for_gemm=use_iris_buffer_for_gemm,
        )
        torch.cuda.synchronize()

        if rank == 0:
            print("[debug] baseline sum:", out_baseline.sum().item())
            print("[debug] custom   sum:", out_custom.sum().item())
            diff = (out_baseline - out_custom).abs().max()
            print("[debug] max |diff|:", diff.item())

        dist.destroy_process_group()
        return

    # M2: warmup + multi timing，split A2A / GEMM
    num_warmup = 10
    num_iters = 10

    # warmup（no timing）
    for _ in range(num_warmup):
        if opt:
            # Custom A2A + grouped GEMM
            CustomA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                shmem,
                False,   # general_a2a
                # default profile False
                use_iris_buffer_for_gemm=use_iris_buffer_for_gemm,
            )
        else:
            # Baseline A2A + GEMM
            TorchA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                False,   # general_a2a
                shmem,
                # default profile False
            )
        shmem.barrier()

    # all rank warmup first then do the timing
    dist.barrier()

    a2a_sum = 0.0
    gemm_sum = 0.0

    for _ in range(num_iters):
        if opt:
            out, timing = CustomA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                shmem,
                False,     # general_a2a
                profile=True,
                use_iris_buffer_for_gemm=use_iris_buffer_for_gemm,
            )
        else:
            out, timing = TorchA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                False,    # general_a2a
                shmem,
                profile=True,
            )

        a2a_sum += timing["a2a_time"]
        gemm_sum += timing["gemm_time"]

        shmem.barrier()

    a2a_avg = a2a_sum / num_iters
    gemm_avg = gemm_sum / num_iters

    if rank == 0:
        impl = "custom" if opt else "baseline"
        print(f"[rank {rank} | {impl}] A2A: {a2a_avg * 1e3:.3f} ms, GEMM: {gemm_avg * 1e3:.3f} ms")

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


    # M1:Only 1 correctness check 
    # Set to True to run a one-shot correctness check (baseline vs custom sums).
    debug_check = False

    if debug_check:
        #all rank run baseline + custom, only print in rank 0 
        out_baseline = TorchA2A(
            rank, tokens, expert_weights, chunk_size,
            batch, seq, hidden_dim, num_experts,
            world_size, False #General A2A 
            , shmem,
        )
        out_custom = CustomA2A(
            rank, tokens, expert_weights, chunk_size,
            batch, seq, hidden_dim, num_experts,
            world_size, shmem, False,#general A2A
        )
        torch.cuda.synchronize()

        if rank == 0:
            #print(
                #f"[rank {rank}] baseline sum = {out_baseline.sum().item():.6f}, "
                #f"custom sum = {out_custom.sum().item():.6f}"
                print("[debug] baseline sum:", out_baseline.sum().item())
                print("[debug] custom   sum:", out_custom.sum().item())
                diff = (out_baseline - out_custom).abs().max()
                print("[debug] max |diff|:", diff.item())

            #)
        dist.destroy_process_group()
        return

    #M2: warmup + Nulti timing, apart A2A / GEMM 

    num_warmup = 10
    num_iters = 10
    
    # warmup without timing
    for _ in range(num_warmup):
        if opt:
            # Custom A2A + grouped GEMM
            CustomA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                shmem,
                False,   # general_a2a
                # profile default False
            )
        else:
            # Baseline A2A + GEMM
            TorchA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                False,   # general_a2a
                shmem,
                # profile default False
            )
        shmem.barrier()

    # timing after all ranks are set
    dist.barrier()

    a2a_sum = 0.0
    gemm_sum = 0.0

    for _ in range(num_iters):
        if opt:
            out, timing = CustomA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                shmem,
                False,    # general_a2a
                profile=True,
            )
        else:
            out, timing = TorchA2A(
                rank,
                tokens,
                expert_weights,
                chunk_size,
                batch,
                seq,
                hidden_dim,
                num_experts,
                world_size,
                False,    # general_a2a
                shmem,
                profile=True,
            )

        a2a_sum += timing["a2a_time"]
        gemm_sum += timing["gemm_time"]

        shmem.barrier()

    a2a_avg = a2a_sum / num_iters
    gemm_avg = gemm_sum / num_iters

    #  all_reduce for the average of all rank
    avg_tensor = torch.tensor([a2a_avg, gemm_avg], device=device)
    dist.all_reduce(avg_tensor, op=dist.ReduceOp.SUM)
    avg_tensor /= world_size

    # rank 0 print average of all rank
    if rank == 0:
        print(
            f"[{'custom' if opt else 'baseline'}][global mean] "
            f"A2A: {avg_tensor[0].item() * 1000:.3f} ms, "
            f"GEMM: {avg_tensor[1].item() * 1000:.3f} ms"
        )

    # loacal average of each rank
    print(
        f"[rank {rank} | {'custom' if opt else 'baseline'}] "
        f"A2A: {a2a_avg * 1000:.3f} ms, GEMM: {gemm_avg * 1000:.3f} ms"
    )

    torch.cuda.synchronize()
    dist.destroy_process_group()
    
    #    diff = (out_baseline - out_custom).abs().max()
    #    if rank == 0:
    #        print(f"[rank 0] max |diff| = {diff.item():.6e}")

    #    return
    # End of correctness check
    
    # Warmup
    #for _ in range(5):
    #    if opt:
    #        CustomA2A(
    #            rank, tokens, expert_weights, chunk_size,
    #            batch, seq, hidden_dim, num_experts,
    #            world_size, shmem, False,
     #       )
    #    else:
    #        TorchA2A(
    #            rank, tokens, expert_weights, chunk_size,
    #            batch, seq, hidden_dim, num_experts,
    #            world_size, False, shmem,
    #        )
    #    shmem.barrier()

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
    #torch.cuda.synchronize()
    #start_time = time.time()
    #for _ in range(10):
    #    if opt:
    #        CustomA2A(
    #            rank, tokens, expert_weights, chunk_size,
    #            batch, seq, hidden_dim, num_experts,
    #            world_size, shmem, False,
    #        )
     #   else:
     #       TorchA2A(
     #           rank, tokens, expert_weights, chunk_size,
     #           batch, seq, hidden_dim, num_experts,
     #           world_size, False, shmem,
     #       )
      #  shmem.barrier()

    #torch.cuda.synchronize()
    #end_time = time.time()
    #print(f'[rank: {rank}] time taken: {(end_time - start_time):.5f}')
    #dist.destroy_process_group()



if __name__ == "__main__":
    ## Input parameters. 

    #The original large configuratio hit a HIP invalid argument due to the grid dimension (grid.y = transmit_size > 65535).
    #For now I used a slightly smaller batch/sequence size so that the per-rank chunk size is under this limit
    world_size, batch, seq, hidden_dim, topk = 8, 4, 1024, 768, 2
    #world_size, batch, seq, hidden_dim, topk = 8, 2, 256, 768, 2

    num_experts = world_size * 32  ## Multiple experts per device, evenly distributed. ##
    run_custom_a2a: bool = True
#    ## A custom test case for convenience. ##
#    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size, topk, run_custom_a2a), nprocs=world_size, join=True)
    ## Generate expert weights once in main (CPU), then broadcast as arg. ##
    use_iris_buffer_for_gemm=True
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