import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.nn.functional as F

from utils import gen_tensor

import triton
import triton.language as tl

import sys
import os

import iris

import time #4 timing

## Experimental preamble to enable a more general and unbalanced a2a. Currently a WIP. ##
@triton.jit
def alltoalldispatch_preamble(
    A, B, META, NUMS_flag, heap_bases,
    cur_rank: tl.constexpr, world_size: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr, EP_SIZE: tl.constexpr):
    """
    This is a fairly non-intuitive kernel. It is meant to emulate an `all_to_all_single` 
    from pytorch (link here: https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single).

    A -> [a1, a2, ...., ak] -> these are the physical values that need to be scattered to the different GPUs.
    B -> [0... 0] 0 initialized array of size world_size that determines how many tokens 
                        should be routed from dev_other -> curr_dev. 
    META -> [m1, m2, ... mf] -> of size: num_experts. This determines the number of tokens routed to expert mi 
                                    from this device.
    NUMS_flag -> This is a unit-sized array which will symbolize when the comms has completed.
    This kernel should be launched with grid-size = NUM_EXPERTS, without this invariant, it will fail.
    """
    pid = tl.program_id(0)
    ## Let's code a little defensively for now. ##
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    ## We can later replace this loop with block-level parallelism: launch world_size blocks. ##
    for device_id in tl.range(world_size):
        ## Here we may to have make a decision as to which
        ##  expert to route to.

        ## First, figure out the device id this expert belongs to.
        chunk_size : tl.constexpr = NUM_EXPERTS // EP_SIZE

        ## Extract the count. ##
        ptrs = tl.arange(0, chunk_size) + chunk_size * device_id
        cnt = tl.sum(tl.load(META + ptrs, mask=ptrs<NUM_EXPERTS))
        
        ## Can we make this better? Use device_ids only and not anything else?
        iris.atomic_add(
            B + cur_rank, 
            cnt,
            cur_rank,
            device_id,
            heap_bases,
            mask=None, ## Should be a legal call since we're not doing anything special here.
            sem="acquire",
            scope="sys"
            )
        iris.atomic_add(
            NUMS_flag,
            1,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="release",
            scope="sys"
        )

    ## Is this needed since we synchronize using shmem.barier() later? ##
    ## Experiments indicate that we do need them. Seems like in-flight 
    ##  values will not be flushed and observed out of the kernel.
    world_size_i32 = tl.full([], world_size, dtype=tl.int32)
    while tl.load(NUMS_flag) != world_size_i32:
        pass

## Currently only supports transmission of fixed, transmit_size packets. ##
@triton.jit
def alltoalldispatch_main(
    routed_token_buffer, input_dev_tokens, 
    DATA_flag, LOCAL_flag, stride_am, stride_ak, heap_bases, 
    token_cnt: tl.constexpr, outgoing_buffer_size: tl.constexpr,
    hidden_dim: tl.constexpr, transmit_size: tl.constexpr, 
    cur_rank: tl.constexpr, world_size: tl.constexpr,
    NUM_EXPERTS: tl.constexpr, EP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr 
):
    """
    This is the main kernel that physically transmits the data over.
    routed_token_buffer -> the array of tokens that we paste into (tokens come from other devices)
    input_dev_tokens -> the array of tokens currently residing on this device.
    DATA_flag -> unit-sized array that will determine when comms have finished.
    LOCAL_flag -> world_size-sized array that will determine when it is safe to increment DATA_flag.
    transmit_size -> number of tokens from Device A -> B per message (we fix this for ease of implementation).
    cur_rank -> self-explanatory.
    """

    ## We need to change the parallelism here to make it amenable for larger scale benchmarking. ##

    pid = tl.program_id(1)
    device_id = tl.program_id(0)

    non_local_ptrs = cur_rank * transmit_size * stride_am + pid * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak

    ptrs = pid * stride_am + device_id * transmit_size * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak 

    for iter in tl.range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        iris.put(
            input_dev_tokens + ptrs,
            routed_token_buffer + non_local_ptrs,
            cur_rank,
            device_id,
            heap_bases,
            mask=tl.arange(0, BLOCK_SIZE)[None, :] + iter * BLOCK_SIZE < hidden_dim, 
            )

        non_local_ptrs += BLOCK_SIZE * stride_ak 
        ptrs += BLOCK_SIZE * stride_ak

    #iris.atomic_add(
    #    LOCAL_flag + device_id,
    #    1,
    #    cur_rank,
    #    cur_rank,
    #    heap_bases,
    #    mask=None,
    #    sem="release",
    #    scope="sys"
    #)

    tl.atomic_add(
        LOCAL_flag + device_id,
        1,
        sem="release"
    )

    if pid == 0:
        ## One block spin-locks until it is safe to atomically increment the global array flag. ##
        #result = 0
        #while result < transmit_size:
        #    compare, value = transmit_size, transmit_size
        #    result = iris.atomic_cas(
        #        LOCAL_flag + device_id,
        #        compare,
        #        value,
        #        cur_rank,
        #        cur_rank,
        #        heap_bases,
        #        sem="acquire",
        #        scope="sys"
        #    ) 

        ## This seems to be buggy. ##
        while tl.load(LOCAL_flag + device_id) != transmit_size:
            pass
    
        ## We have to incr the atomic flag only once at the end once the shift is successful. ##
        iris.atomic_add(
            DATA_flag,
            1,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="release",
            scope="sys" 
        )

def grouped_triton_gemm(
    routed_token_buffer: torch.Tensor,
    expert_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Minimal grouped GEMM baseline in pure PyTorch.

    routed_token_buffer: [N_tokens, hidden_dim]
    expert_weights:      [num_experts, hidden_dim, expert_dim]

    For now we assume that tokens for each expert are laid out in contiguous
    chunks along dim=0, each chunk having the same length.
    """
    assert routed_token_buffer.dim() == 2
    assert expert_weights.dim() == 3

    num_experts, hidden_dim_w, expert_dim = expert_weights.shape
    total_tokens, hidden_dim_x = routed_token_buffer.shape
    assert hidden_dim_x == hidden_dim_w, "Hidden dim mismatch between tokens and expert weights"

    assert total_tokens % num_experts == 0, "Total tokens must be divisible by num_experts"
    tokens_per_expert = total_tokens // num_experts

    outputs = []
    offset = 0
    for i in range(num_experts):
        x_i = routed_token_buffer[offset: offset + tokens_per_expert]   # [n_i, hidden_dim]
        w_i = expert_weights[i]                                         # [hidden_dim, expert_dim]
        y_i = x_i @ w_i                                                 # [n_i, expert_dim]
        outputs.append(y_i)
        offset += tokens_per_expert

    return torch.cat(outputs, dim=0)     # [N_tokens, expert_dim]


#def run(
#    rank: int, tokens: torch.tensor, transmit_size: int, batch: int, seq: int, hidden_dim: int, num_experts: int,
#    world_size: int, shmem, general_a2a: bool 
#):
#    """
#    This is the callee function for the Shmem-based all-to-all + gemm kernels.

#    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
#    transmit_size: num_experts sized tensor representing the number of tokens routed to expert_i.
#    general_a2a: Flag to trigger unbalanced a2a capability, currently not working.
#    """
def run(
    rank: int,
    tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    transmit_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    shmem,
    general_a2a: bool,
    # timing profile switch
    profile: bool = False
):
    device_id = rank % torch.cuda.device_count()

    if general_a2a:
        ## Instantiate shmem based heap regions over here. ##
        #NUMS_flag = shmem.zeros(1, dtype=torch.int32, device="cuda")
        #device_cnts = shmem.zeros(world_size, device="cuda")

        ## First, we have to call an alltoall that will aggregrate token level information
        ##   to instantiate buffer sizes.
        #alltoalldispatch_preamble[(1,1,1)](
        #    tokens, device_cnts, meta, NUMS_flag, shmem.get_heap_bases(), 
        #    dist.get_rank(), world_size, num_experts, tokens.shape[0], world_size
        #) 

        ## Next, we instantiate token buffers accordingly for the next phase of the all-to-all + gemm. ##
        #routed_token_buffer = shmem.zeros(int(round(device_cnts.sum().item())), tokens.shape[-1])
        raise NotImplementedError("general_a2a path not implemented yet.")  
    else:
        # Fixed-size balanced all-to-all path.

        # Ensure tokens are on the correct device.
        tokens = tokens.to(f"cuda:{device_id}")
        
        # Shmem-based receive buffer.
        ## For now, we fix a fix sized buffer to transmit over, otherwise things get far too complicated.
        assert tokens.shape[0] % world_size == 0, 'Tensor sizes not properly shaped.'
        routed_token_buffer = shmem.zeros(transmit_size * world_size, tokens.shape[-1], dtype=tokens.dtype, device="cuda")
        #assert routed_token_buffer.shape[-1] == tokens.shape[-1], 'incorrect tensor sizes for source/dest buffers.'
        
        # Flags for synchronization between ranks. 
        DATA_flag = shmem.zeros(world_size, device="cuda")
        LOCAL_flag = torch.zeros(world_size, dtype=torch.int32).to(tokens.device)
        #outgoing_buffer_size = transmit_size * world_size

        # 2streams A2A/GEMM    
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        # A2A(Trition + SHMEM) Timing START
        if profile:
            torch.cuda.synchronize()
            t0 = time.time()
    ## Now, we launch the main all-to-all kernel + persistent gemm. ##
        with torch.cuda.stream(s1):
            alltoalldispatch_main[(world_size,transmit_size,1)](
                routed_token_buffer, tokens, DATA_flag, LOCAL_flag,
                tokens.stride(0), tokens.stride(1), shmem.get_heap_bases(), 
                tokens.shape[0], transmit_size * world_size, 
                tokens.shape[-1], transmit_size, rank, 
                world_size, num_experts, world_size, BLOCK_SIZE=256  ## Temporarily put 256 but autotune out in the future.
            )


        # After the end of kernel, do barrier again, ensure A2a completion
        torch.cuda.synchronize()
        shmem.barrier()  # 重新启用：其开销算在 A2A 内

        if profile:
            a2a_time = time.time() - t0
        # End of A2A timing

        if rank == 0:
            print("[custom] routed_token_buffer sum =", routed_token_buffer.sum().item())
        
        # Grouped GEMM timing start
        if profile:
            torch.cuda.synchronize()
            t1 = time.time()
        
        
        
        #with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        NUM_REM_SMS = 100 - world_size

    #   shmem.barrier() ## -> It seems like this is a requirement? Why is this the case, I am locally synchronizing.?
        #print(f'[rank: {rank}], summed tensor: {routed_token_buffer.sum()}, num_ranks: {shmem.get_num_ranks()}')

            ## Call the grouped GEMM here that does the expert MLP compute. ##
        with torch.cuda.stream(s2):           
            #NUM_REM_SMS = 100 - world_size
            moe_output = grouped_triton_gemm(
                routed_token_buffer,
                expert_weights,
            )
        # Ensure both streams have finished before returning.
        torch.cuda.synchronize()
        if profile:
            gemm_time = time.time() - t1
        # End of Group gemm timing
        # Ensure both streams have finished before returning.
        torch.cuda.synchronize()

        if profile:
            return moe_output, {
                "a2a_time": a2a_time,
                "gemm_time": gemm_time,
            }

        return moe_output