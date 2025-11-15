import torch
import torch.nn.functional as F
import torch.distributed as dist
import math

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