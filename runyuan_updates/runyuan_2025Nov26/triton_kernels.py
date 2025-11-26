# triton_kernels.py

import torch
import triton
import triton.language as tl


# ================================
# 1. 标准 2D matmul kernel: Y = X @ W
# ================================

@triton.jit
def matmul_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    标准单个 GEMM：X: [M, K], W: [K, N] -> Y: [M, N]
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + offs_k

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offsets[None, :] * stride_xk
        w_ptrs = W_ptr + k_offsets[:, None] * stride_wk + offs_n[None, :] * stride_wn

        x_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        w_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(y_ptrs, acc, mask=y_mask)


def triton_gemm(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    X: [M, K]
    W: [K, N]
    返回 Y: [M, N]
    """
    M, K = X.shape
    K2, N = W.shape
    assert K == K2

    Y = torch.empty((M, N), dtype=X.dtype, device=X.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

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
# 2. grouped matmul kernel: 多个 expert 一起算
# ================================

@triton.jit
def grouped_matmul_kernel(
    X_ptr,            # [total_tokens, H]
    W_flat_ptr,       # [num_experts * H, D] (把 W.view(...) 后的扁平矩阵)
    Y_ptr,            # [total_tokens, D]
    total_tokens,     # = tokens_per_expert * num_experts
    hidden_dim,       # H
    expert_dim,       # D
    tokens_per_expert,
    stride_xm, stride_xk,
    stride_wm, stride_wn,
    stride_ym, stride_yn,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    对 num_experts 个 expert 做分组 GEMM：

      对于每个 expert e:
          X_e: [tokens_per_expert, H]
          W_e: [H, D]
          Y_e: [tokens_per_expert, D]

    X 在物理上是 [total_tokens, H]，按照 expert 块排布：
        [X_0; X_1; ...; X_{E-1}]
    W_flat 是 [num_experts * H, D]，把 [E, H, D] 拉平成 [E*H, D]。
    Y 同 X，一样的分块。
    """

    expert_id = tl.program_id(0)       # 哪个 expert
    pid_m = tl.program_id(1)           # 这个 expert 内部的第几个 M-block

    # 当前 expert 的 token 行号区间： [e * T, (e+1) * T)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_idx = expert_id * tokens_per_expert + offs_m   # 全局行索引

    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, hidden_dim, BLOCK_K):
        k_offsets = k + offs_k

        # --- 读 X ---
        x_ptrs = (
            X_ptr
            + row_idx[:, None] * stride_xm
            + k_offsets[None, :] * stride_xk
        )
        x_mask = (row_idx[:, None] < total_tokens) & (k_offsets[None, :] < hidden_dim)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # --- 读 W_flat ---
        # 对应这一 expert 内的行号：
        #   row_in_W = expert_id * hidden_dim + k_offsets
        w_row = expert_id * hidden_dim + k_offsets
        w_ptrs = (
            W_flat_ptr
            + w_row[:, None] * stride_wm
            + offs_n[None, :] * stride_wn
        )
        w_mask = (w_row[:, None] < num_experts * hidden_dim) & (offs_n[None, :] < expert_dim)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    # --- 写回 Y ---
    y_ptrs = (
        Y_ptr
        + row_idx[:, None] * stride_ym
        + offs_n[None, :] * stride_yn
    )
    y_mask = (row_idx[:, None] < total_tokens) & (offs_n[None, :] < expert_dim)
    tl.store(y_ptrs, acc, mask=y_mask)


def grouped_triton_gemm(
    routed_token_buffer: torch.Tensor,   # [total_tokens, H]
    W: torch.Tensor,                     # [num_experts, H, D]
) -> torch.Tensor:
    """
    一次性对所有 expert 做 Y = X_e @ W_e：

    routed_token_buffer: [E * T, H]，其中 E = num_experts, T = tokens_per_expert
    W: [E, H, D]
    返回 Y: [E * T, D]
    """
    total_tokens, hidden_dim = routed_token_buffer.shape
    num_experts, H2, expert_dim = W.shape
    assert H2 == hidden_dim, f"W.shape[1] ({H2}) must equal hidden_dim ({hidden_dim})"
    assert total_tokens % num_experts == 0, "total_tokens 必须能整除 num_experts"

    tokens_per_expert = total_tokens // num_experts

    # [E, H, D] -> [E * H, D]
    W_flat = W.view(num_experts * hidden_dim, expert_dim).contiguous()

    Y = torch.empty(
        (total_tokens, expert_dim),
        dtype=routed_token_buffer.dtype,
        device=routed_token_buffer.device,
    )

    stride_xm, stride_xk = routed_token_buffer.stride()
    stride_wm, stride_wn = W_flat.stride()
    stride_ym, stride_yn = Y.stride()

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (
        num_experts,
        triton.cdiv(tokens_per_expert, BLOCK_M),
    )

    grouped_matmul_kernel[grid](
        routed_token_buffer,
        W_flat,
        Y,
        total_tokens,
        hidden_dim,
        expert_dim,
        tokens_per_expert,
        stride_xm, stride_xk,
        stride_wm, stride_wn,
        stride_ym, stride_yn,
        num_experts=num_experts,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return Y
