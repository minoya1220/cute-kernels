import cutlass
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

    M, N = gX.shape

    m = bidx
    n_iters = N // bdim
    n = tidx

    # single pass online safe softmax
    accum = 0.0
    max = -Float32.inf
    for ni in range(0, n_iters):
        # load and do the reduction
        prev_max = max
        elem = gX[bidx, n_iters * bdim + tidx]
        max = cute.math.max(elem, max)
        accum = accum * cute.math.exp(prev_max - max) + cute.math.exp(elem - max)

    # TODO: write reduction at block scale, use butterfly reduction

    # TODO: write elementwise finisher for softmax


    pass


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
    torch_softmax = torch.nn.Softmax()
    print(x)
    print(torch_softmax(x, dim=0))