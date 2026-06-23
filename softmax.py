import cutlass
import operator
import torch
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
from cutlass.cute.runtime import from_dlpack
# Online Safe Softmax algorithm
# first pass: reduction - find sum(e^(x-running_max))
# second pass: elementwise ops, do e^(x-max) / first_pass_sum
# assume f32 to start make more general later
# gX and gY are 2D row-major tensors M x N where M is the batch dimension and N is the contiguous dim that softmax is applied to 
# 
BLOCK_SIZE = 256

@cute.kernel
def softmax_fwd_kernel(gX: cute.Tensor, gY: cute.Tensor, cX: cute.Tensor, tv_layout: cute.Layout, tiler: cute.Shape, shape: cute.Shape):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim = BLOCK_SIZE # , _, _ = cute.arch.block_dim() # ensures that this value is known at compile time
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    
    smem = cutlass.utils.SmemAllocator()
    buffer_layout = cute.make_layout(cute.arch.WARP_SIZE)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    blk_coord = (bidx, None)
    blkX = cute.local_tile(gX, tiler, blk_coord) # divides the block into tiles, boundary checks still have to be included
    blkY = cute.local_tile(gY, tiler, blk_coord)
    cblkX = cute.local_tile(cX, tiler, blk_coord)
    
    frgX = cute.composition(blkX, tv_layout)
    frgY = cute.composition(blkY, tv_layout)
    cfrgX = cute.composition(cblkX, tv_layout)

    thr_coord = (tidx, None)
    thrX = frgX[thr_coord]
    thrY = frgY[thr_coord]
    cthrX = cfrgX[thr_coord]

    # initialize SMEM buffers
    if widx == 0:
        reduction_buffer[lidx] = 0.0
        max_buffer[lidx] = -Float32.inf
    cute.arch.sync_threads()

    M, N = shape

    m = bidx
    n_iters = cute.ceil_div(N, bdim)
    rPred = cute.make_fragment(cthrX.shape, cutlass.Boolean)
    rX = cute.make_rmem_tensor_like(thrX)

    # single pass online safe softmax
    accum = 0.0
    thread_max = -Float32.inf
    for ni in range(0, n_iters):
        # load and do the reduction
        rPred[ni] = cute.elem_less(cthrX[ni], shape)
        rX[ni] = thrX[ni].load() if rPred[ni] else -Float32.inf
        prev_max = thread_max
        thread_max = cute.arch.fmax(rX[ni], thread_max)
        accum = accum * cute.math.exp(prev_max - thread_max) + cute.math.exp(rX[ni] - thread_max)

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
        if rPred[ni]:
            elem = rX[ni]
            thrY[ni] = cute.math.exp(elem - full_max) / divisor


@cute.jit
def softmax_fwd_launcher(mX, mY):
    # mX is the input tensor, mY is the output
    bdim = BLOCK_SIZE
    shape = mX.shape
    M, N = shape

    n_iters = cute.ceil_div(N, bdim)

    thr_layout = cute.make_layout((bdim,))
    val_layout = cute.make_layout((n_iters,), (bdim,))
    # use cute.recast_layout to make this work for arbitrary dtypes
    tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    
    gX = mX # cute.zipped_divide(mY, tiler=tiler)
    gY = mY # cute.zipped_divide(mX, tiler=tiler)
    cX = cute.make_identity_tensor(mX.shape) # cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler=tiler)

    fwd_kernel = softmax_fwd_kernel(gX, gY, cX, tv_layout, tiler, shape)
    fwd_kernel.launch(
        grid=(cute.size(mX.shape[:-1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1)
    )


@cute.kernel
def softmax_bwd_kernel(gY: cute.Tensor, gdY: cute.Tensor, gdX: cute.Tensor):
    
    pass


@cute.jit
def softmax_bwd_launcher(T):
    pass


if __name__ == "__main__":
    X = torch.randn(2**6,2**10, device='cuda', dtype=torch.float32)
    Y = torch.randn(2**6,2**10, device='cuda', dtype=torch.float32)

    x_ = from_dlpack(X, assumed_align=16)
    y_ = from_dlpack(Y, assumed_align=16)


    torch_softmax = torch.softmax(X, dim=1)
    softmax_fwd_ = cute.compile(softmax_fwd_launcher, X, Y)
    softmax_fwd_(X, Y)
    torch.testing.assert_close(torch_softmax, Y)
    print("success")