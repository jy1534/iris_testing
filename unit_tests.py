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

def gen_gemm_input(num_local_experts, token_hid_dim, expert_hid_dim):
    expert_token_cnt = torch.randint(low=0, high=100, size=(num_local_experts,), device="cpu")
    tokens = torch.randn(int(expert_token_cnt.sum().item()), token_hid_dim)
    weights = torch.randn(num_local_experts, token_hid_dim, expert_hid_dim)
    return tokens, weights, expert_token_cnt

def test_gemm(total_expert_cnt, token_hid_dim, expert_hid_dim):
    tokens, weights, expert_token_cnt = gen_gemm_input(total_expert_cnt, token_hid_dim, expert_hid_dim)
    custom_output = expert(tokens, weights, expert_token_cnt, total_expert_cnt)

    torch_out = []
    tokens_seen = 0
    for i in range(total_expert_cnt):
        n = int(expert_token_cnt[i].item())
        torch_out.append(torch.einsum('sd,df->sf',
                                      tokens[tokens_seen:tokens_seen+n, :],
                                      weights[i]))
        tokens_seen += n

    # 这里到底应该 cat 还是 stack，取决于 expert(...) 的输出布局
    # 如果 custom_output 是把所有 expert 的输出按 token 顺序拼起来，应该用 cat：
    ref = torch.cat(torch_out, dim=0)
    return is_correct(custom_output, ref, 1e-2)

if __name__ == '__main__':

    ## Some sample inputs to test out correctness. ##
    test_gemm(2, 24, 48)
    test_gemm(5, 128, 128)

    

