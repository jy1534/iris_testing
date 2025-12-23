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

# Global SHMEM buffer cache to avoid heap OOM 

_routed_token_buffer_full_cache = None
_routed_token_buffer_full_cache_shape = None
_routed_token_buffer_full_cache_dtype = None

_global_sum_cache = None
_global_sum_cache_shape = None


def _get_chunk_buffers(shmem, world_size, transmit_size, hidden_dim, dtype):
    """
    Reuse large SHMEM buffers across multiple CustomA2A() calls.

    We only allocate once per shape+dtype, and then zero_() them in-place
    instead of calling shmem.zeros(...) every time, to avoid bumping the
    IRIS heap pointer repeatedly (which causes Heap out of memory).
    """
    global _routed_token_buffer_full_cache
    global _routed_token_buffer_full_cache_shape
    global _routed_token_buffer_full_cache_dtype
    global _global_sum_cache
    global _global_sum_cache_shape

    needed_shape_full = (world_size, transmit_size, hidden_dim)
    needed_shape_sum = (1,)

    need_new_full = (
        _routed_token_buffer_full_cache is None
        or _routed_token_buffer_full_cache_shape != needed_shape_full
        or _routed_token_buffer_full_cache_dtype != dtype
    )

    if need_new_full:
        # First time (or shape/dtype changed): allocate from heap
        _routed_token_buffer_full_cache = shmem.zeros(
            needed_shape_full,
            dtype=dtype,
        )
        _routed_token_buffer_full_cache_shape = needed_shape_full
        _routed_token_buffer_full_cache_dtype = dtype
    else:
        # Reuse existing buffer: just reset content
        _routed_token_buffer_full_cache.zero_()

    need_new_sum = (
        _global_sum_cache is None
        or _global_sum_cache_shape != needed_shape_sum
    )

    if need_new_sum:
        _global_sum_cache = shmem.zeros(
            needed_shape_sum,
            dtype=torch.float32,
        )
        _global_sum_cache_shape = needed_shape_sum
    else:
        _global_sum_cache.zero_()

    return _routed_token_buffer_full_cache, _global_sum_cache



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
    NUM_EXPERTS: tl.constexpr, EP_SIZE: tl.constexpr,
    # new for chunking
    pid_offset: tl.constexpr, pid_count: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    routed_token_buffer: global recv buffer, shape [world_size * transmit_size, hidden_dim]
    input_dev_tokens:    send buffer on this rank, same shape
    transmit_size:       global slots per peer (full_transmit_size)
    pid_offset:          global slot offset for this kernel launch
    pid_count:           number of slots for this launch (= grid.y)
    """

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

    #pid = tl.program_id(1)
    pid_local = tl.program_id(1)  # 0 .. pid_count-1
    device_id = tl.program_id(0)

    # global slot id for ths kernel
    pid = pid_offset + pid_local   # 0 .. transmit_size-1


    non_local_ptrs = cur_rank * transmit_size * stride_am + pid * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak

    ptrs = pid * stride_am + device_id * transmit_size * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak 

    offs_k = tl.arange(0, BLOCK_SIZE)[None, :]

    for iter in tl.range(tl.cdiv(hidden_dim, BLOCK_SIZE)):
        mask = offs_k + iter * BLOCK_SIZE < hidden_dim

        iris.put(
            input_dev_tokens + ptrs,
            routed_token_buffer + non_local_ptrs,
            cur_rank,
            device_id,
            heap_bases,
            mask=mask,
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

    #if pid == 0:
    if pid_local == 0:
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
        #while tl.load(LOCAL_flag + device_id) != transmit_size:
        while tl.load(LOCAL_flag + device_id) != pid_count:
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

    routed_token_buffer: [N_tokens, hidden_dim]        (2D)
                         or [world_size, transmit, hidden_dim] (3D, will be flatten)
    expert_weights:      [num_experts, hidden_dim, expert_dim]
    """

   # Added: 3D buffer with chunk path support
    if routed_token_buffer.dim() == 3:
        world_size, transmit_size, hidden_dim = routed_token_buffer.shape
        routed_token_buffer = routed_token_buffer.view(world_size * transmit_size,
                                                       hidden_dim)
    else:
        assert routed_token_buffer.dim() == 2, \
            f"Unexpected routed_token_buffer.dim() = {routed_token_buffer.dim()}"

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
    profile: bool = False,
    # iris or not
    use_iris_buffer_for_gemm: bool = True
):
    device_id = rank % torch.cuda.device_count()

    if general_a2a:
        raise NotImplementedError(...)
    else:
        # grid.y of the cluster  ~ 65535
        MAX_GRID_Y_NOCHUNK = 60000

        full_transmit_size = transmit_size

        if full_transmit_size <= MAX_GRID_Y_NOCHUNK:
            # Route smaller transmit_size to the legacy non-chunked path (original implementation).
            return _run_balanced_nochunk(
                rank, tokens, expert_weights, transmit_size,
                batch, seq, hidden_dim, num_experts,
                world_size, shmem, profile,
                use_iris_buffer_for_gemm,
            )
        else:
            # only configs greater than grid.y go chunk
            print(f"[rank {rank}] using chunked path, transmit_size={full_transmit_size}")
            return _run_balanced_chunked(
                rank, tokens, expert_weights, transmit_size,
                batch, seq, hidden_dim, num_experts,
                world_size, shmem, profile,
                use_iris_buffer_for_gemm,
            )

        # After the end of kernel, do barrier again, ensure A2a completion
        #torch.cuda.synchronize()
        #shmem.barrier()  # reactivate cost in A2A 

        #if profile:
            #a2a_time = time.time() - t0
        # End of A2A timing

        #if rank == 0:
            #print("[custom] routed_token_buffer sum =", routed_token_buffer.sum().item())
        
        # Grouped GEMM timing start
        #if profile:
            #torch.cuda.synchronize()
            #t1 = time.time()
        
        
        
        #with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        #NUM_REM_SMS = 100 - world_size

    #   shmem.barrier() ## -> It seems like this is a requirement? Why is this the case, I am locally synchronizing.?
        #print(f'[rank: {rank}], summed tensor: {routed_token_buffer.sum()}, num_ranks: {shmem.get_num_ranks()}')

            ## Call the grouped GEMM here that does the expert MLP compute. ##
        #with torch.cuda.stream(s2):           
            #NUM_REM_SMS = 100 - world_size
            #moe_output = grouped_triton_gemm(
                #routed_token_buffer,
                #expert_weights,
            #)
        # Ensure both streams have finished before returning.
        #torch.cuda.synchronize()
        #if profile:
            #gemm_time = time.time() - t1
        # End of Group gemm timing
        # Ensure both streams have finished before returning.
        #torch.cuda.synchronize()

        #if profile:
            #return moe_output, {
                #"a2a_time": a2a_time,
                #"gemm_time": gemm_time,
            #}

        #return moe_output


# 4degungging
def _run_balanced_chunked(
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
    profile: bool,
    # new iris part
    use_iris_buffer_for_gemm: bool,
):
    device_id = rank % torch.cuda.device_count()
    tokens = tokens.to(f"cuda:{device_id}")
    tokens = tokens.contiguous()

    assert tokens.shape[0] % world_size == 0, "Tensor sizes not properly shaped."

    hidden_dim = tokens.shape[-1]
    full_transmit_size = transmit_size
    total_tokens = tokens.shape[0]
    assert total_tokens == full_transmit_size * world_size, \
        "tokens.shape[0] must equal transmit_size * world_size"

    # HIP limit：grid.y < ~65535
    # for debug 1024 before -> too many chunks -> 89ms
    MAX_GRID_Y = 60000   # or 65000
    max_chunk = min(full_transmit_size, MAX_GRID_Y)

    # Global receive buffer: to be fed into grouped GEMM.
    routed_token_buffer_full, global_sum = _get_chunk_buffers(
        shmem=shmem,
        world_size=world_size,
        transmit_size=transmit_size,
        hidden_dim=hidden_dim,
        dtype=tokens.dtype,
    )

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    tokens_view = tokens.view(world_size, full_transmit_size, hidden_dim)

    if profile:
        torch.cuda.synchronize()
        t0 = time.time()

    # Suggestion: Initialize DATA_flag and LOCAL_flag once outside the while loop, and simply .zero_() them for each chunk
    DATA_flag = shmem.zeros(world_size, device="cuda")
    LOCAL_flag = torch.zeros(world_size, dtype=torch.int32, device=tokens.device)

    with torch.cuda.stream(s1):
        remaining = full_transmit_size
        pid_start = 0

        while remaining > 0:
            curr_chunk = min(remaining, max_chunk)

            # Zero out flags before processing each chunk.
            DATA_flag.zero_()
            LOCAL_flag.zero_()

            if rank == 0:
                chunk_tokens = tokens_view[:, pid_start:pid_start + curr_chunk, :]
                print(
                    f"[chunk debug] BEFORE kernel: pid_start={pid_start}, "
                    f"curr_chunk={curr_chunk}, "
                    f"chunk_tokens_sum={chunk_tokens.sum().item()}"
                )

            alltoalldispatch_main[(world_size, curr_chunk, 1)](
                routed_token_buffer_full,
                tokens,
                DATA_flag,
                LOCAL_flag,
                tokens.stride(0),
                tokens.stride(1),
                shmem.get_heap_bases(),
                tokens.shape[0],
                full_transmit_size * world_size,
                hidden_dim,
                full_transmit_size,
                rank,
                world_size,
                num_experts,
                world_size,
                pid_start,           # pid_offset
                curr_chunk,          # pid_count
                BLOCK_SIZE=256,
            )

            if rank == 0:
                print(
                    f"[chunk debug] AFTER kernel: pid_start={pid_start}, "
                    f"curr_chunk={curr_chunk}, "
                    f"global_sum_now={routed_token_buffer_full.sum().item()}"
                )

            pid_start += curr_chunk
            remaining -= curr_chunk

    torch.cuda.synchronize()
    shmem.barrier()

    if profile:
        a2a_time = time.time() - t0

    if rank == 0 and not profile:
        print("[custom/chunk] routed_token_buffer_full sum =",
              routed_token_buffer_full.sum().item())

    # GEMM
    if profile:
        torch.cuda.synchronize()
        t1 = time.time()
    
    # new iris switch
    if use_iris_buffer_for_gemm:
        gemm_input = routed_token_buffer_full          # directly calculate on IRIS buffer 
        if rank == 0 and profile:
            print("[GEMM] using IRIS buffer")
    else:
        # clone one part to CUDA allocator 
        gemm_input = routed_token_buffer_full.clone()
        if rank == 0 and profile:
            print("[GEMM] using cloned CUDA buffer")
    with torch.cuda.stream(s2):
            moe_output = grouped_triton_gemm(
                gemm_input,
                expert_weights,
            )

    torch.cuda.synchronize()
    if profile:
        gemm_time = time.time() - t1
        return moe_output, {
            "a2a_time": a2a_time,
            "gemm_time": gemm_time,
        }

    return moe_output


def _run_balanced_nochunk(
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
    profile: bool,
    # new iris 
    use_iris_buffer_for_gemm: bool,
):
    """
    fixed all-to-all + grouped GEMM
    """
    device_id = rank % torch.cuda.device_count()

    # ensure ranks on gpu and persistent
    tokens = tokens.to(f"cuda:{device_id}")
    tokens = tokens.contiguous()

    # tokens.shape[0] = world_size * transmit_size
    assert tokens.shape[0] % world_size == 0, "Tensor sizes not properly shaped."
    hidden_dim_local = tokens.shape[-1]

    # recevie buffer on symmetric heap of Iris
    routed_token_buffer = shmem.zeros(
        transmit_size * world_size,
        hidden_dim_local,
        dtype=tokens.dtype,
        device="cuda",
    )
    assert routed_token_buffer.shape[-1] == hidden_dim_local, \
        "incorrect tensor sizes for source/dest buffers."

    DATA_flag = shmem.zeros(world_size, device="cuda")
    LOCAL_flag = torch.zeros(world_size, dtype=torch.int32, device=tokens.device)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # A2A timing
    if profile:
        torch.cuda.synchronize()
        t0 = time.time()

    with torch.cuda.stream(s1):
        alltoalldispatch_main[(world_size, transmit_size, 1)](
            routed_token_buffer,
            tokens,
            DATA_flag,
            LOCAL_flag,
            tokens.stride(0),
            tokens.stride(1),
            shmem.get_heap_bases(),
            tokens.shape[0],
            transmit_size * world_size,
            hidden_dim_local,
            transmit_size,
            rank,
            world_size,
            num_experts,
            world_size,
            0,                 # pid_offset = 0
            transmit_size,     # pid_count = transmit_size
            BLOCK_SIZE=256,
        )

    # waiting for all rank complete A2A
    torch.cuda.synchronize()
    shmem.barrier()

    if profile:
        a2a_time = time.time() - t0

    if rank == 0 and not profile:
        print("[custom/nochunk] routed_token_buffer sum =",
              routed_token_buffer.sum().item())
    
    # grouped GEMM timing 
    if profile:
        torch.cuda.synchronize()
        t1 = time.time()

    # new iris switch

    if use_iris_buffer_for_gemm:
        gemm_input = routed_token_buffer
        if rank == 0 and profile:
            print("[GEMM] using IRIS buffer (nochunk)")
    else:
        gemm_input = routed_token_buffer.clone()
        if rank == 0 and profile:
            print("[GEMM] using cloned CUDA buffer (nochunk)")

    with torch.cuda.stream(s2):
        moe_output = grouped_triton_gemm(
            gemm_input,     
            expert_weights,
        )

    torch.cuda.synchronize()
    if profile:
        gemm_time = time.time() - t1
        return moe_output, {"a2a_time": a2a_time, "gemm_time": gemm_time}

    return moe_output
