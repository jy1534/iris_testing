import torch
import torch.distributed as dist

from .kernels import token_shuffle


class TokenShuffle(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        ## Local buffers. 
        dispatched_tokens, pca
        ## Synchronization variables
        token_sync
    ):
    """
    Reshuffles tokens from uneven all-to-all to eliminate zero-padding.

    Args:
        ctx: context used for pytorch bwd/fwd.
        dispatched_tokens (Tensor): [E, world_size, capacity, hidden_dim]-sized tensor
            that stores the dispatched tokens (after all-to-all shuffling).
        pca (Tensor): [E, world_size]-sized physical_counts_array tensor that determines
            number of incoming tokens routed to the current rank for an expert. 
            pca[i, j] = x indicates device j is routing x tokens to expert i on the current rank.
        token_sync (Tensor): [E]-sized array indicating synchronization variables for when the current
            tokens have successfully routed to expert i on the current rank.
    """

        ## First, we aggregrate token counts. 
        ## This will be metadata the token-shuffling
        ##  operation will consume. 
        cum_summed_tkn_cnt = pca.cumsum()
        rolled_tkn_cnt = torch.roll(pca.view(-1).cumsum(), shifts=1)
        rolled_tkn_cnt[0] = 0
        rolled_tkn_cnt = rolled_tkn_cnt.view(e_local, -1)

        # Next, we create a packed output buffer.
        total_tkn_cnt = cum_summed_tkn_cnt[-1]
        shuffled_tokens = torch.zeros((total_tkn_cnt, hidden_dim), dtype=send_payload.dtype).to(send_payload.device)

        ## Finally, launch token shuffling kernel. ##
        grid = (e_local, world_size, buffers.token_buf.size(2))

        token_shuffle[grid](
            cum_summed_tkn_cnt, pca,
            dispatched_tokens, 
            shuffled_tokens,
            token_sync,
            e_local, world_size,
            dispatched_tokens.size(2), dispatched_tokens.size(-1),
            128
        ) 

        return shuffled_tokens

    @staticmethod
    def backward(ctx,):
        pass

shuffle = TokenShuffle.apply