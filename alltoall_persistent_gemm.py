import torch
import torch.nn.functional as F
import torch.distributed as dist

import triton
import triton.language as tl

import iris  # Triton 侧 kernel 里直接用 iris.* 原语
# 注意：如果你把 triton_gemm / grouped_triton_gemm 放在 triton_kernels.py 里，
# 可以改成：from triton_kernels import triton_gemm, grouped_triton_gemm


# ================================
# Triton kernels（基于 Ahan 原版）
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
    WIP: emulate torch.distributed.all_to_all_single 风格的“统计用”预处理，
    用于 general_a2a（不等长 all-to-all）场景。

    A:      没有真正用到，只是沿用原接口
    B:      [world_size]，统计每个 device/rank 上 token 数
    META:   [NUM_EXPERTS]，每个 expert 的路由计数
    NUMS_flag: 标记所有 device 完成的计数
    """
    pid = tl.program_id(0)
    tl.device_assert(NUM_EXPERTS % EP_SIZE == 0)

    # 每个 device_id 一次，累加 META 中属于该 device 的路由计数
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

    # 简单 spin 等待所有 device 完成
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
    主 all-to-all kernel：
      - 从本 rank 的 input_dev_tokens 中，按块拷贝到 routed_token_buffer 中对应位置。
      - world_size × transmit_size grid:
          program_id(0) = device_id   (目标 device/rank)
          program_id(1) = pid         (第几个“消息块”)
    """

    pid = tl.program_id(1)        # 第几个 token block
    device_id = tl.program_id(0)  # 目标 device (rank)

    # non_local_ptrs: 当前 rank 在 routed_token_buffer 中的起始位置
    non_local_ptrs = (
        cur_rank * transmit_size * stride_am
        + pid * stride_am
        + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak
    )

    # ptrs: 本 rank input_dev_tokens 中对应 block 的位置
    ptrs = (
        pid * stride_am
        + device_id * transmit_size * stride_am
        + tl.arange(0, BLOCK_SIZE)[None, :] * stride_ak
    )

    # 沿 hidden_dim 方向分块拷贝
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

    # 本地计数：对 LOCAL_flag[device_id] 做一个 atomic_add
    tl.atomic_add(
        LOCAL_flag + device_id,
        1,
        sem="release",
    )

    if pid == 0:
        # 等待该 device_id 的所有 blocks 完成（共 transmit_size 次）
        while tl.load(LOCAL_flag + device_id) != transmit_size:
            pass

        # 全部完成后，用 Iris 的跨 rank atomic 通知对方
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
# （可选）小 helper：Triton GEMM 包装
# 如果你已经在 triton_kernels.py 里实现了 triton_gemm / grouped_triton_gemm
# 这里可以不要；反之可以保留一份简单版本。
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
# Python 侧入口：run()
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
    Shmem all-to-all + Phase1 FFN GEMM（当前是 grouped GEMM 版本）。

    Args:
        rank:         本进程 rank
        tokens:       [num_tokens, hidden_dim] 本 rank 输入 token
        transmit_size: 每个 rank -> 其他 rank 的 token 数（均匀 case）
        batch, seq, hidden_dim: 形状元信息（目前主要用于 sanity check）
        num_experts:  expert 个数
        world_size:   总 rank 数
        shmem:        ShmemCompat 实例（基于 Iris）
        general_a2a:  是否走不均匀 all-to-all path（目前未实现）
    """

    device_id = rank % max(torch.cuda.device_count(), 1)

    # ============================
    # (1) allocate buffers
    # ============================

    if general_a2a:
        raise NotImplementedError(
            "general_a2a=True 路径目前未在 Iris 版本中实现；"
            "请先使用 general_a2a=False 跑均匀 all-to-all + GEMM。"
        )
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

    # ============================
    # (3) Phase1: grouped Triton GEMM FFN
    # ============================

    from utils import gen_expert_weights
    from triton_kernels import grouped_triton_gemm  # 确保有这个函数

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
