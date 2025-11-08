import torch.multiprocessing as mp

"""
A lot of this code is taken from: 
https://github.com/ROCm/iris/blob/main/examples/07_gemm_all_scatter/gemm_all_scatter.py
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

import triton
import triton.language as tl
from examples.common.utils import read_realtime

import sys
import os

import iris
"""
First exchange token size data.

## We set number of blocks to number of devices to send the information to.

Suppose we have the following experts:
Ex1, Ex2, Ex3, Ex4.

1 & 2 on one device.

3 & 4 on one device.

We should launch two blocks.

On device 0:
block 1 transmits the information to "local" (alt. also recieves stuff locally).

block 2 transmits the information to device 2 (alt. also recieves stuff locally).

We must use iris.atomic_adds for convenience.


NUMS_global -> array consisting of integers that determine the number of tokens routed to an expert from this TB (dev.).
"""
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
    META -> [m1, m2, ... mf] -> this is the cumulative sum of the number of tokens that need to be routed to an expert.
                                    Its size is NUM_EXPERTS + 1.
    NUMS_flag -> This is an array of world_size (0 initialized) as flags to understand when comms have finished. 
    This kernel should be launched with grid-size = NUM_EXPERTS, without this invariant, it will fail.
    """
    pid = tl.program_id(0)
    ## Let's code a little defensively for now. ##
    tl.device_assert(NUM_EXPERTS == tl.num_programs(0))
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    for exp in tl.range(NUM_EXPERTS):
        ## Here we may to have make a decision as to which
        ##  expert to route to.

        ## First, figure out the device id this expert belongs to.
        device_id = exp // (NUM_EXPERTS // EP_SIZE)

        ## Extract the count. ##
        ptrs = tl.arange(2)+pid
        
        meta_vals = tl.load(META+ptrs,mask=ptrs<N+1)

        total_cnt = meta_vals[1] - meta_vals[0]

        ## Can we make this better? Use device_ids only and not anything else?
        if device_id == cur_rank:
            tl.atomic_add(B+device_id, total_cnt)
            tl.atomic_add(NUMS_flag, 1)
        else:
            iris.atomic_add(
                NUMS_global + pid, 
                total_cnt,
                cur_rank,
                device_id,
                heap_bases,
                mask=None ## Should be a legal call since we're not doing anything special here.
                )
            iris.atomic_add(
                NUMS_flag,
                1,
                cur_rank,
                device_id,
                heap_bases,
                mask=None
            )

    while tl.atomic_cas(NUMS_flag, world_size, world_size + 1) != world_size + 1:
        pass

def alltoalldispatch_main(
    A, B, META, DATA_flag, stride_am, stride_ak
):
    """
    This is the main kernel that physically transmits the data over.
    A -> vectors to transmit over.
    B -> empty array zero-initialized to plunk all the data into.
    """
    pass

@triton.jit
def persistent_gemm_all_scatter(
    A,
    B,
    C,
    c_global,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_cm_global,
    stride_cn_global,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    heap_bases: tl.tensor,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    COLLECT_TIMESTAMPS: tl.constexpr = False,
    mm_begin_timestamp_ptr: tl.tensor = None,
    mm_end_timestamp_ptr: tl.tensor = None,
):
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_min(mm_begin_timestamp_ptr + tile_id, timestamp)

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        # Accumulator registers with C results
        c = acc.to(C.type.element_ty)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # Add compiler hints
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # Define the C-mask (BLOCK_SIZE_M, 1) x (1, BLOCK_SIZE_N)
        sub_mask = (rm[:, None] < M) & (rn[None, :] < N)

        # Calculate the "global" offset of C based on the rank.
        # Note how the N-dimension is being multiplied by current rank.
        # This is because each rank is computing a portion of the N-dimension
        # locally and then scattering it to all other ranks to complete
        # the global N-dimension.
        global_offset = rm[:, None] * stride_cm_global + (rn[None, :] + cur_rank * N) * stride_cn_global

        # Timestamp for GEMM before store
        if COLLECT_TIMESTAMPS:
            timestamp = read_realtime()
            tl.atomic_max(mm_end_timestamp_ptr + tile_id, timestamp)

        # Store data to the global result using puts
        for remote_rank in range(world_size):
            if remote_rank == cur_rank:
                # For the current rank, we can use store
                tl.store(c_global + global_offset, c, mask=sub_mask)
            else:
                iris.store(
                    c_global + global_offset,
                    c,
                    cur_rank,
                    remote_rank,
                    heap_bases,
                    mask=sub_mask,
                )

def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int) -> tuple[torch.tensor, torch.tensor]:
    torch.manual_seed(rank)
    tokens = torch.randn(batch*seq, hidden_dim).to("cuda" if torch.cuda.is_available() else "cpu")

    ## We first do a common load tensor. ##
    assert (batch*seq) % num_experts == 0, 'must be evenly divisible'
    meta = torch.tensor([(batch*seq) // num_experts for _ in range(num_experts)]).to("cuda" if torch.cuda.is_availabel() else "cpu")

    return tokens, meta

def callee(
    batch: int, seq: int, hidden_dim: int, num_experts: int,
    world_size: int
    ):
    """
    This is the callee function for the Shmem-based all-to-all + gemm kernels.

    Tokens: [cnt, hidden dimension]-sized tensor representing the input tokens to this layer.
    meta: num_experts sized tensor representing the number of tokens routed to expert_i.
    """
    heap_size = 2**30 ## 1 GiB symmetric heap.
    shmem = iris.iris(heap_size)
    rank = dist.get_rank()
    tokens, meta = gen_tensor(batch, seq, hidden_dim, world_size, num_experts, rank)

    device_cnts = torch.zeros(world_size).to(tokens.device)
    meta_cumsum = torch.cumsum(F.pad(meta, (1, 0), "constant", 0), dim=0)

    ## Instantiate shmem based heap regions over here. ##
    NUMS_flag = shmem.zeros(num_experts)

    ## First, we have to call an alltoall that will aggregrate token level information
    ##   to instantiate buffer sizes.
    alltoalldispatch_preamble[(num_experts,1,1)](
        tokens, device_cnts, meta_cumsum, NUMS_flag, shmem.get_heap_bases(), 
        dist.get_rank(), world_size, num_experts, tokens.shape[0], world_size // num_experts
    ) 

    ## Let's print device_cnts at the end. ##
    print(device_cnts)

    ## Next, we instantiate token buffers accordingly for the next phase of the all-to-all + gemm.
    routed_token_buffer = shmem.zeros(device_cnts.sum(), tokens.shape[-1])
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    ## Now, we launch the main all-to-all kernel + persistent gemm.
    with torch.cuda.stream(s1):
        ## Call the main all-to-all over here that transmits the data. ##
        pass

    with torch.cuda.stream(s2):
        ## Call the persistent Gemm here that does the MLP compute. ##
        pass

if __name__ == "__main__":
    ## Input parameters. ##
    world_size, batch, seq, hidden_dim = 2, 2, 2, 4  
    num_experts = world_size * 2
    ## A custom test case for convenience. ##
    mp.spawn(callee, args=(batch, seq, hidden_dim, num_experts, world_size, ), nprocs=world_size, join=True)