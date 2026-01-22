import torch
import torch.distributed as dist

from .layers.all_to_all import custom_a2a
from .layers.token_shuffle import shuffle
from .layers.expert import expert


## Some simple testing utility functions.

def is_correct(one, two, threshold):
    if one.shape != two.shape:
        return abs(one.sum() - two.sum()) < threshold

    return torch.allclose(one, two, rtol=threshold)

def gen_tensor(shape, cuda=True):
    return torch.randn(*shape, dtype=torch.bfloat16).to("cuda" if cuda else "cpu")


def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0,high=100, (num_local_experts,))

    tokens = torch.randn(expert_token_cnt.sum(), token_hid_dim)

    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)

    return tokens, weights

def test_gemm(total_expert_cnt, token_hid_dim, expert_hid_dim):
    
    tokens, weights = gen_gemm_input(total_expert_cnt, token_hid_dim, expert_hid_dim)

    custom_output = expert(tokens, weights,expert_token_cnt,total_expert_cnt)

    ## We use pytorch as ground truth. ##
    torch_out = []
    tokens_seen = 0
    for i in range(total_expert_cnt):
        torch_out.append(torch.einsum('sd,df->sf', tokens[tokens_seen:expert_token_cnt[i], :], weights[i]))
        tokens_seen += expert_token_cnt[i]

    return is_correct(custom_output, torch.stack(torch_out), 1e-2)

if __name__ == '__main__':

    ## Some sample inputs to test out correctness. ##
    test_gemm(2, 24, 48)
    test_gemm(5, 128, 128)

    

