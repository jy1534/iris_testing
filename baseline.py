"""
Baseline a2a implementation to compare perf against.
"""

import torch
import torch.distributed as dist

def run(
    rank: int, tokens: torch.tensor, chunk_size: int, 
    batch: int, seq: int, hidden_dim: int, 
    num_experts: int, world_size: int, 
    general_a2a: bool, shmem
):
    if general_a2a:
        pass ## Not implemented yet. ##
    else:
        tokens_recv = torch.zeros(chunk_size * world_size, hidden_dim, dtype=tokens.dtype).to("cuda" if torch.cuda.is_available() else "cpu")
        dist.all_to_all_single(tokens_recv, tokens)

        torch.cuda.synchronize()
        #print(f'[rank: {rank}], summed token buffer: {sum([i.sum() for i in tokens_recv])}')