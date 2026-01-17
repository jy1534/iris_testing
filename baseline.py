import torch
import torch.distributed as dist


def run_padding_free(
    *,
    rank: int,
    world_size: int,
    send_payload: torch.Tensor,            # [sum_send, H]
    send_counts: torch.Tensor,             # [world, E_local] int32
    counts_all: torch.Tensor,              # [world, world, E_local] int32 (src, dst, e)
    local_weights: torch.Tensor,           # [E_local, H, O]
) -> torch.Tensor:
    """Reference (baseline) padding-free pipeline using NCCL all_to_all_single.

    This matches the *semantics* of the IRIS implementation:
      1) counts_all is assumed already exchanged (metadata stage)
      2) payload stage uses variable split sizes (true alltoallv)
      3) grouped GEMM runs after the payload stage completes (no overlap)

    Output layout also matches the IRIS consumer kernel layout
    """
    assert send_payload.is_cuda and local_weights.is_cuda
    assert send_counts.dtype == torch.int32
    assert counts_all.dtype == torch.int32

    # compute split sizes 
    # input splits are per-destination rows
    input_split_sizes = send_counts.sum(dim=1).to(torch.int64).tolist()  # len=world

    # output splits are per-source rows destined for 'rank'
    recv_counts = counts_all[:, rank, :]  # [src, E_local]
    output_split_sizes = recv_counts.sum(dim=1).to(torch.int64).tolist()  # len=world

    total_recv = int(sum(output_split_sizes))
    hidden_dim = send_payload.shape[1]
    out_dim = local_weights.shape[2]
    e_local = local_weights.shape[0]

    # payload all-to-all (variable) 
    recv_payload = torch.empty((total_recv, hidden_dim), device=send_payload.device, dtype=send_payload.dtype)

    # the "second all-to-all" 
    dist.all_to_all_single(
        recv_payload,
        send_payload,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    # grouped GEMM reference

    # Layout of recv_payload is src-major and within each src, expert-local-major.
    recv_counts_i64 = recv_counts.to(torch.int64)
    expert_offsets = torch.cumsum(recv_counts_i64, dim=1) - recv_counts_i64  # [src, e]

    src_sizes = recv_counts_i64.sum(dim=1)  # [src]
    src_base = torch.cumsum(src_sizes, dim=0) - src_sizes  # [src]

    out = torch.empty((e_local, total_recv, out_dim), device=send_payload.device, dtype=send_payload.dtype)
    out.zero_()

    # Compute per block. This is a reference; correctness > speed.
    for src in range(world_size):
        sbase = int(src_base[src].item())
        for e in range(e_local):
            n = int(recv_counts_i64[src, e].item())
            if n == 0:
                continue
            off = int(expert_offsets[src, e].item())
            rows = slice(sbase + off, sbase + off + n)
            x = recv_payload[rows, :]  # [n, H]
            w = local_weights[e, :, :]  # [H, O]
            out[e, rows, :] = x @ w

    return out
