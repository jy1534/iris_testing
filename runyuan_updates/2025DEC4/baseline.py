"""
Baseline a2a implementation to compare perf against.
"""

import torch
import torch.distributed as dist

#def run(
#    rank: int, tokens: torch.tensor, chunk_size: int, 
#    batch: int, seq: int, hidden_dim: int, 
#    num_experts: int, world_size: int, 
#    general_a2a: bool, shmem
#):
def run(
    rank: int,
    tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    chunk_size: int,
    batch: int,
    seq: int,
    hidden_dim: int,
    num_experts: int,
    world_size: int,
    general_a2a: bool,
    shmem,
):
    if general_a2a:
        pass ## Not implemented yet. ##
    else:
        # 1) Baseline all-to-all using PyTorch collectives.
        tokens_recv = torch.zeros(
            chunk_size * world_size,
            hidden_dim,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        dist.all_to_all_single(tokens_recv, tokens)
        if rank == 0:
            print("[baseline] tokens_recv sum =", tokens_recv.sum().item())

        torch.cuda.synchronize()
        
        # 2) Baseline GEMM: simple per-expert sequential matmul, matching
        #    the semantics of grouped_triton_gemm.
        assert expert_weights.dim() == 3
        num_experts_w, hidden_dim_w, expert_dim = expert_weights.shape
        total_tokens, hidden_dim_x = tokens_recv.shape
        assert num_experts_w == num_experts, "num_experts mismatch in baseline"
        assert hidden_dim_x == hidden_dim_w, "hidden dim mismatch in baseline"

        assert total_tokens % num_experts == 0, "Total tokens must be divisible by num_experts"
        tokens_per_expert = total_tokens // num_experts

        outputs = []
        offset = 0
        for i in range(num_experts):
            x_i = tokens_recv[offset: offset + tokens_per_expert]   # [n_i, hidden_dim]
            w_i = expert_weights[i]                                 # [hidden_dim, expert_dim]
            y_i = x_i @ w_i                                         # [n_i, expert_dim]
            outputs.append(y_i)
            offset += tokens_per_expert

        moe_output = torch.cat(outputs, dim=0)   # [total_tokens, expert_dim]
        torch.cuda.synchronize()
        return moe_output