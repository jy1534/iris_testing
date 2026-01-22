import torch
import torch.nn.functional as F

from .kernels import grouped_gemm

class Expert(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        # Local buffers 
        tokens, weights,
        local_expert_cnt,
        aggregrate_exp_cnt
    ):
    """
    Expert GeMM compute stage. This layer: Computes a single layer MLP
    for all local experts.

    Args:
        ctx: context required for torch fwd/bwd pass.
        tokens (Tensor): [S, hidden_dim]-sized tensor. S is the packed size of the tensor
            (without zero-padding).
        weights (Tensor): [E, hidden_dim, ffn_hidden_dim]-sized tensor representation MLP 
            weights per expert.
        local_expert_cnt (Tensor): [E]-sized tensor representing the total number of tokens
            corresponding to expert i on the current rank.
        aggregrate_exp_cnt (int): the total number of experts across all devices.
    """

        local_expert_count = torch.Tensor(1).to(int32) + (expert_cnt // dist.get_world_size)

        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        NUM_SM = 142

        if torch.is_cuda():
            NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count

        gemm_grid = (NUM_SM,)

        processed_tokens = torch.zeros((tokens.shape[0], weights.size(-1)), dtype=tokens.dtype).to(tokens.device)
        # Finally, launch the grouped-gemm kernel.
        grouped_gemm[gemm_grid](
            tokens,
            weights, 
            processed_tokens, # Shape: [S, expert_hidden_dim]
            local_expert_cnt, # Shape: [E], token count per expert.
            NUM_SM,
            expert_cnt // dist.get_world_size(),
            tokens.size(-1),
            weights.size(-1),
            # tile sizes
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )

    return processed_tokens 

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError('Backward pass not implemented yet.')

expert = Expert.apply