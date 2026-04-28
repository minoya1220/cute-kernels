import cutlass
import operator
import torch
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
# Online Safe Softmax algorithm
# first pass: reduction - find sum(e^(x-running_max))
# second pass: elementwise ops, do e^(x-max) / first_pass_sum
# assume f32 to start make more general later
# gX and gY are 2D row-major tensors M x N where M is the batch dimension and N is the contiguous dim that softmax is applied to 
# 
@cute.kernel
def softmax_fwd_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    
    smem = cutlass.utils.SmemAllocator()
    buffer_layout = cute.make_layout(cute.arch.WARP_SIZE)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    # initialize SMEM buffers
    if widx == 0:
        reduction_buffer[lidx] = 0.0
        max_buffer[lidx] = -Float32.inf
    cute.arch.sync_threads()

    M, N = gX.shape

    m = bidx
    n_iters = cute.ceil_div(N, bdim)

    # single pass online safe softmax
    accum = 0.0
    thread_max = -Float32.inf
    for ni in range(0, n_iters):
        # load and do the reduction
        prev_max = thread_max
        elem = gX[bidx, ni * bdim + tidx] if ni * bdim + tidx < N else -Float32.inf
        thread_max = cute.arch.fmax(elem, thread_max)
        accum = accum * cute.math.exp(prev_max - thread_max) + cute.math.exp(elem - thread_max)

    # intra-warp reduction
    warp_max = cute.arch.warp_reduction(thread_max, cute.arch.fmax)
    accum = accum * cute.math.exp(thread_max - warp_max)
    accum = cute.arch.warp_reduction(accum, operator.add)
    
    # write each warp result to SMEM
    if lidx == 0:
        reduction_buffer[widx] = accum
        max_buffer[widx] = warp_max
    cute.arch.sync_threads()

    # inter-warp reduction
    partial_max = max_buffer[lidx] 
    partial_result = reduction_buffer[lidx] 
    full_max = cute.arch.warp_reduction(partial_max, cute.arch.fmax)
    partial_result = partial_result * cute.math.exp(partial_max - full_max)
    divisor = cute.arch.warp_reduction(partial_result, operator.add)

    # elementwise finisher
    for ni in range(0, n_iters):
        idx = ni * bdim + tidx
        if idx < N:
            elem = gX[bidx ,idx]
            gY[bidx ,idx] = cute.math.exp(elem - full_max) / divisor


@cute.jit
def softmax_fwd_launcher(T):
    block_size = 256

    pass

@cute.kernel
def softmax_bwd_kernel(gY: cute.Tensor, gdY: cute.Tensor, gdX: cute.Tensor):
    
    pass


@cute.jit
def softmax_bwd_launcher(T):
    pass


if __name__ == "__main__":
    x = torch.randn(2**5)
    torch_softmax = torch.nn.Softmax(dim=0)
    print(x)
    print(torch_softmax(x, dim=0))