import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from adapter import ShmemCompat
from alltoall_persistent_gemm import run


def main():

    # -----------------------------
    # Fix:intilization of fake distributed
    # no env variable
    # -----------------------------
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="file:///tmp/iris_test_dist_init",
            world_size=1,
            rank=0,
        )

    world_size = 1
    rank = 0

    batch, seq, hidden_dim = 4, 4, 512
    num_experts = 4

    device = "cuda"

    tokens = torch.randn(batch * seq, hidden_dim, dtype=torch.bfloat16, device=device)
    transmit_size = tokens.shape[0]

    # Shmem
    shmem = ShmemCompat()

    out = run(
        rank=rank,
        tokens=tokens,
        transmit_size=transmit_size,
        batch=batch,
        seq=seq,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        world_size=world_size,
        shmem=shmem,
        general_a2a=False,
    )

    print("Output shape:", out.shape)


if __name__ == "__main__":
    main()
