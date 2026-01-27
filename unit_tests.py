

import os
import torch.multiprocessing as mp

from runtime.orchestrator import CaseConfig, run_case

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "2"))

def _worker(rank: int):
    case = CaseConfig(
        batch=2,
        seq=128,
        hidden_dim=64,
        topk=2,
        e_local=4,
        capacity=1024,
        seed=123,
    )
    res = run_case(rank=rank, world_size=WORLD_SIZE, case=case, mode="correctness")
    # Basic assertions on rank 0
    if rank == 0:
        assert res["check_pca"]["pca_max_abs_diff"] == 0, res
        assert res["check_token_buf"]["token_buf_valid_max_abs_diff"] == 0.0, res
        print("[PASS]", res["check_pca"], res["check_token_buf"])

if __name__ == "__main__":
    mp.spawn(_worker, nprocs=WORLD_SIZE, join=True)
