
import os
import torch.multiprocessing as mp

from runtime.orchestrator import CaseConfig, run_case

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "8"))

def _worker(rank: int):
    case = CaseConfig(
        batch=int(os.getenv("BATCH", "4")),
        seq=int(os.getenv("SEQ", "2048")),
        hidden_dim=int(os.getenv("HIDDEN", "4096")),
        topk=int(os.getenv("TOPK", "2")),
        e_local=int(os.getenv("E_LOCAL", "4")),
        capacity=int(os.getenv("CAPACITY", "32768")),
        seed=int(os.getenv("SEED", "42")),
    )
    mode = os.getenv("MODE", "both")  # perf/correctness/both
    res = run_case(rank=rank, world_size=WORLD_SIZE, case=case, mode=mode)
    if rank == 0:
        print(res)

if __name__ == "__main__":
    mp.spawn(_worker, nprocs=WORLD_SIZE, join=True)
