import torch
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

import iris  
##Some imports has been deleted for they seems not to be utilized 

# ================================
# Triton kernels（based on the original one）
# ================================

@triton.jit
def alltoalldispatch_preamble(
    A,
    B,
    META,
    NUMS_flag,
    heap_bases,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    N: tl.constexpr,
    EP_SIZE: tl.constexpr,
):
    """
    WIP: emulate torch.distributed.all_to_all_single pre-processing for stats，
    for general_a2a

    A:      not really utilized
    B:      [world_size]，calculate each tokens of device/rank 
    META:   [NUM_EXPERTS]，count of experts on each routing
    NUMS_flag: label the finish of device
    """
    pid = tl.program_id(0)
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    # once for each device_id ,accumulate the routing count belongs to that device of the META 
    for device_id in tl.range(world_size):
        chunk_size: tl.constexpr = NUM_EXPERTS // EP_SIZE
        ptrs = tl.arange(0, chunk_size) + chunk_size * device_id
        cnt = tl.sum(tl.load(META + ptrs, mask=ptrs < NUM_EXPERTS))

        iris.atomic_add(
            B + cur_rank,
            cnt,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="acquire",
            scope="sys",
        )
        iris.atomic_add(
            NUMS_flag,
            1,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="release",
            scope="sys",
        )

    #  ez spin wait for the completion of all
    world_size_i32 = tl.full([], world_size, dtype=tl.int32)
    while tl.load(NUMS_flag) != world_size_i32:
        pass


@triton.jit
def alltoalldispatch_main(
    routed_token_buffer,
    input_dev_tokens,
    DATA_flag,
    LOCAL_flag,
    stride_am,
    stride_ak,
    heap_bases,
    token_cnt: tl.constexpr,
    outgoing_buffer_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    transmit_size: tl.constexpr,
    cur_rank: tl.constexpr,
    world_size: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Main all-to-all kernel:
      - Performs a block-wise copy from the current rank's `input_dev_tokens` to the corresponding position in `routed_token_buffer`.
      - Grid configuration (world_size x transmit_size):
          program_id(0) = device_id   (Target device/rank)
          program_id(1) = pid         (Message block index)
    """


    pid = tl.program_id(1)        # which token block
    device_id = tl.program_id(0)  # target device (rank)

    # non_local_ptrs: routed_token_buffer the initial position of currnt rank
    non_local_ptrs = (
        cur_rank * transmit_size * stride_am
        + pid * stride_am
        + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak
    )

    # ptrs:Position of the corresponding block in the current rank's input_dev_tokens."

    ptrs = (
        pid * stride_am
        + device_id * transmit_size * stride_am
        + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak
    )

    # Block-wise copy along hidden_dim

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

    # Local count: Perform atomic_add on LOCAL_flag[device_id]

    tl.atomic_add(
        LOCAL_flag + device_id,
        1,
        sem="release",
    )

    if pid == 0:
      # Wait for all blocks for this device_id to complete (total: transmit_size)

        while tl.load(LOCAL_flag + device_id) != transmit_size:
            pass

    # Upon completion, notify the target rank using Iris cross-rank atomic

        iris.atomic_add(
            DATA_flag,
            1,
            cur_rank,
            device_id,
            heap_bases,
            mask=None,
            sem="release",
            scope="sys",
        )


# ================================
# Little helper of the Python-side Triton GEMM
# ================================

def triton_gemm(X, W):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2

    Y = torch.empty((M, N), dtype=X.dtype, device=X.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        (M + BLOCK_M - 1) // BLOCK_M,
        (N + BLOCK_N - 1) // BLOCK_N,
    )

    from triton_kernels import matmul_kernel  # 如果你单独放在这个文件

    matmul_kernel[grid](
        X, W, Y,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return Y


# ================================
# Python entrance：run()
# ================================

def run(
    rank: int,
    tokens: torch.Tensor,
    transmit_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    shmem,
    general_a2a: bool,
):
   """
    Shmem all-to-all + Phase1 FFN GEMM (currently grouped GEMM version).

    Args:
        rank:         Current process rank.
        tokens:       [num_tokens, hidden_dim] Input tokens for the current rank.
        transmit_size: Number of tokens sent from each rank to every other rank (uniform case).
        batch, seq, hidden_dim: Shape metadata (currently used mainly for sanity checks).
        num_experts:  Total number of experts.
        world_size:   Total number of ranks (world size).
        shmem:        ShmemCompat instance (based on Iris).
        general_a2a:  Whether to use the non-uniform all-to-all path (currently not implemented).
"""


    device_id = rank % max(torch.cuda.device_count(), 1)

    # ============================
    # (1) allocate buffers
    # ============================

    if general_a2a:
        raise NotImplementedError(
            "general_a2a=True NO"##not implemented 
            
        )
    ##The original "WIP non-uniform all-to-all path" has been disabled and now explicitly raises an exception. This prevents potential bugs such as undefined metadata.
    ##Only the uniform all-to-all path is retained to simplify testing; the general path can be re-implemented in the future.
    ##The current Iris all-to-all implementation only supports uniform partitioning. This code explicitly enforces this constraint on the caller.

    else:
        assert (
            tokens.shape[0] % world_size == 0
        ), f"Tensor sizes not properly shaped: {tokens.shape[0]} vs world_size={world_size}"

        routed_token_buffer = shmem.zeros(
            transmit_size * world_size,
            tokens.shape[-1],
            dtype=tokens.dtype,
        )

        assert routed_token_buffer.shape[-1] == tokens.shape[-1]

        DATA_flag = shmem.zeros(world_size, dtype=torch.int32)
        LOCAL_flag = torch.zeros(
            world_size, dtype=torch.int32, device=tokens.device
        )
        ##Removed explicit device='cuda' argument in shmem.zeros. It now relies on shmem's default device placement (which should align with the current rank's CUDA device, assuming ShmemCompat handles this correctly).
        ##Explicitly set DATA_flag dtype to torch.int32. While int was likely the default previously, this is now explicit for clarity
        ##LOCAL_flag is initialized with device=tokens.device directly in the constructor.
        ##Potentially avoids minor overhead from .to(...) copies, depending on the default device behavior of torch.zeros.
        ##Integration: Relies more heavily on shmem.zeros semantics. Better？
        ##Explicitly setting DATA_flag to int32 more aligns with the requirements for Triton kernels and atomic operations.
        outgoing_buffer_size = transmit_size * world_size
        
    # ============================
    # (2) launch all-to-all kernel
    # ============================

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

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
            tokens.shape[-1],
            transmit_size,
            rank,
            world_size,
            num_experts,
            world_size,
            BLOCK_SIZE=256,
        )

    with torch.cuda.stream(s2):
        NUM_REM_SMS = 100 - world_size  # 占位

    shmem.barrier()
    ##The current version does not yet achieve true overlap between all-to-all communication and GEMM. Task (1) merely establishes the s1/s2 stream infrastructure, but the actual computation still executes on the default stream
    ##In the future, the Phase 1 GEMM block can be moved into a with torch.cuda.stream(s2): context to align with Iris's stream binding mechanisms.

    # ============================
    # (3) Phase1: grouped Triton GEMM FFN
    # ============================
    ##"After the all-to-all communication completes, invoke grouped_triton_gemm on the routed_token_buffer to execute the first-stage FFN grouped GEMM."
    ##Initialize the expert weight matrix W on the Python side using gen_expert_weights(). This is a direct implementation of Task (2).
    ##Requires total_tokens % num_experts == 0. This corresponds to the naive path where every expert receives an equal number of tokens.
    ##Make run more than communication, with moe_output return
    from utils import gen_expert_weights
    from triton_kernels import grouped_triton_gemm  # make sure this one exist

    device = tokens.device
    dtype = tokens.dtype
    hidden_dim = tokens.shape[-1]
    expert_dim = hidden_dim

    W = gen_expert_weights(
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        expert_dim=expert_dim,
        device=device,
        dtype=dtype,
    )

    total_tokens = routed_token_buffer.shape[0]
    assert (
        total_tokens % num_experts == 0
    ), f"Naive Phase1 requires equal tokens per expert, got {total_tokens} / {num_experts}"

    moe_output = grouped_triton_gemm(routed_token_buffer, W)

    print(f"[rank {rank}] Phase1 grouped GEMM complete, output={moe_output.shape}")

    return moe_output
