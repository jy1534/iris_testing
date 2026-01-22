import torch
import torch.distributed as dist
from .kernels import counts_exchange_kernel, tokens_exchange_kernel, build_expert_offsets

class AllToAllOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        # Local buffers used to store/aggregrate results. 
        # These buffers are potentially written to by other
        # devices.
        tokens, local_pca, 
        # Local buffers that contain messages that will be
        # exchanged to other devices.
        dest_counts 
        # Synchronization variables for Shmem correctness.
        counts_sync, tokens_sync,
        # Heap bases pointing to symmetric memory arrays of
        # all devices participating in EP.
        heap_bases,
        ## Local variables used
        experts,
        capacity
    ):
    """
    Custom Shmem-based AllToAll for Expert parallelism.
    This implements a two-stage AllToAll.

    Args:
        ctx: context used to store data for bwd pass.
        local_pca (Tensor): local physical counts array
            ([E, world_size]). Used to aggregrate token
            counts. local_pca[i, j] = x means device j
            will write x tokens belonging to expert i 
            in the local rank.
        dest_counts (Tensor): [world_size, E]-sized tensor
            dest_counts[k, j] = x means the local rank will
            set device k's local_pca[j, i] = x. 
        counts_sync (Tensor): [1] sized variable used to synchronize
            the completion of stage 1 with stage 2.
        tokens_sync (Tensor): [E]-sized tensor to synchronize the completion
            of stage 2 with potential post kernels (expert compute).
        heap_bases (list[Tensor]): A list containing the addresses 
            of devices' symmetric heaps.
        experts (int): the total number of experts.
        dst_offsets (Tensor): [world_size]-sized tensor. This is metadata
            required for the physical token exchange kernel.
        capacity (int): integer for maximum token count that is routable
            to an expert on a device.
    """

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        assert experts % world_size, 'EP-group size unevenly calculated.'

        counts_exchange_kernel[(world_size,)](
            dest_counts,
            local_pca,
            counts_sync,
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=experts // world_size,
            BLOCK_E=128,
            num_warps=4,
        )

        # Instantiate output buffer.
        output_buffer = torch.zeros((experts // world_size, world_size, capacity, tokens.shape[-1]))

        # Step-2: exchange tokens.
        tokens_exchange_kernel[(world_size, experts // world_size)](
            counts_sync,
            tokens,
            dest_counts,
            dst_offsets,
            build_expert_offsets(dest_counts), ## Is this correct?
            output_buffer,
            tokens_sync,
            heap_bases,
            src_rank=rank,
            world_size=world_size,
            e_local=experts // world_size,
            CAP=capacity,
            hidden_dim=tokens.shape[-1],
            EXPECTED=False, # What on earth is this? No documentation on this whatsover... Please remove...
            BLOCK_M=32,
            BLOCK_K=128,
            num_warps=8,
        )

        return output_buffer

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backwards pass not implemented yet")


custom_a2a = AllToAllOp.apply
 
