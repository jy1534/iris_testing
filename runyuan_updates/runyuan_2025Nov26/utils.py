import torch
import torch.nn.functional as F
import torch.distributed as dist
import math

###new from ry
def gen_expert_weights(
    num_experts: int,
    hidden_dim: int,
    expert_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Generate MoE expert weight matrices.
    Shape: [num_experts, hidden_dim, expert_dim]
    Using Xavier-uniform initialization.
    """
    W = torch.empty(
        num_experts, hidden_dim, expert_dim,
        device=device, dtype=dtype
    )
    torch.nn.init.xavier_uniform_(W)
    return W
##Originally, the file only contained gen_tensor, which was responsible for generating random tokens, performing Top-k routing, and assembling coalesced_experts along with per-device token counts。
##Added gen_expert_weights to specifically generate the MoE expert weight tensor W with shape [num_experts, hidden_dim, expert_dim], utilizing Xavier-uniform initialization
##This is a purely additive entry point. It does not alter the behavior of gen_tensor but provides weight initialization for upstream code (e.g., the Phase 1 GEMM in run())
##Initialization occurs on the Python side using torch.nn.init.xavier_uniform_, which is a standard practical strategy for FFN weight initialization
##This involves a one-time $O(\text{num\_experts} \times \text{hidden\_dim} \times \text{expert\_dim})$ tensor allocation and initialization. The performance impact depends on call frequency. Currently, it is called inside run() (generating a new W per run), but reuse/caching should be considered for future optimization.
##gen_expert_weights does not directly depend on Iris or Triton; it simply provides input weights for the Triton GEMM / grouped GEMM
##In the previous alltoall_shmem.py file, the workflow is established as: import gen_expert_weights inside run(), call W = gen_expert_weights(...), and pass W to grouped_triton_gemm(routed_token_buffer, W).
##This directly addresses Task (2) (assigned by Ahan): initialize the expert weight matrix on the Python side and use it for the subsequent Triton GEMM.

###

def gen_tensor(
    batch: int, seq: int, hidden_dim: int, 
    world_size: int, num_experts: int, 
    rank: int, topk: int) -> tuple[torch.tensor, torch.tensor]:

    torch.manual_seed(rank)
    assert num_experts % world_size == 0, "Incorrect EP_SIZE, world_size should evenly divide num_experts."

    tokens = torch.rand(batch*seq, hidden_dim, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    #router = torch.randn(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Initialization option: 1 -> not super good. ##
    #router = torch.randn(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu") * 0.1
    #router = router + torch.log(torch.tensor(1.0 / num_experts))

    ## Initialization option: 2 -> not super good. ##
    #std = 0.01 / math.sqrt(hidden_dim)
    #router = torch.randn(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu") * std
    #uniform_bias = math.log(1.0 / num_experts)
    #router = router + uniform_bias

    ## Initialization option: 3 -> seems to work the best, but still not ideal. ##
    router = torch.zeros(hidden_dim, num_experts, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
    # Add small random perturbation
    router = router + torch.randn_like(router) * 0.001

    routed_values = F.softmax(torch.einsum('bh, he -> be', tokens, router), dim=-1)
    top_vals, top_idxs = torch.topk(routed_values, topk)

    ## This is very incorrect when it comes to chunking. ## --> Something is going wrong here.
    expert_tokens = []
    for ex in range(num_experts):
        mask = (top_idxs == ex).any(dim=-1)
        tkns = tokens[mask] # (num tokens routed to an expert, hidden dimension).
        if tkns.numel() > 0:
            expert_tokens.append(tkns)
        else:
            tmp_tkns = torch.zeros(1, hidden_dim, dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")
            expert_tokens.append(tmp_tkns)

    ## Next, we pad to the largest element-sized tensor.
    max_tkn_cnt = max([i.shape[0] for i in expert_tokens])

    ## We exchange this between devices. ##
    global_max = torch.tensor([max_tkn_cnt], dtype=torch.int32).to("cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(global_max, dist.ReduceOp.MAX)
    if rank == 0:
        ## Why is the % incorrect? ##
        print(f'[rank: {rank}]: global_max_tkn_cnt: {global_max.item()}, % total tokens: {(global_max.item() / (batch * seq * world_size * topk))*100:.2f}%')

    expert_tokens = [torch.cat(
        (i, torch.zeros(global_max.item() - i.shape[0], hidden_dim, dtype=tokens.dtype).to(tokens.device))
    ) for i in expert_tokens]

    ## Lastly, we have to coalesce for the all-to-all. ##
    coalesced_experts = torch.cat(expert_tokens, dim=0)

    expert_per_device = num_experts // world_size

    return coalesced_experts, global_max.item()*expert_per_device 