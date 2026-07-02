import cutlass
import operator
import torch
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Boolean, const_expr
from cutlass.cute.runtime import from_dlpack
from math import gcd
# Online Safe Softmax algorithm
# first pass: reduction - find sum(e^(x-running_max))
# second pass: elementwise ops, do e^(x-max) / first_pass_sum
# assume f32 to start make more general later
# 
BLOCK_SIZE = 256

@cute.kernel
def softmax_fwd_kernel(
                    gX: cute.Tensor, 
                    gY: cute.Tensor, 
                    cX: cute.Tensor, 
                    tv_layout: cute.Layout, 
                    tiler: cute.Shape, 
                    shape: cute.Shape,
                    tiled_copy: cute.TiledCopy, 
                    n_iters: cutlass.Constexpr, 
                    vec_size: cutlass.Constexpr,
    ):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim = BLOCK_SIZE # , _, _ = cute.arch.block_dim() # ensures that this value is known at compile time
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    
    smem = cutlass.utils.SmemAllocator()
    buffer_layout = cute.make_layout(cute.arch.WARP_SIZE)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    if (tidx == 0 and bidx == 0):
        print(cute.zipped_divide(gX, tiler).layout)

    blk_coord = (bidx, None)
    blkX = cute.local_tile(gX, tiler, blk_coord) # divides the block into tiles, boundary checks still have to be included
    blkY = cute.local_tile(gY, tiler, blk_coord)
    cblkX = cute.local_tile(cX, tiler, blk_coord)
    
    
    if (tidx == 0 and bidx == 0):
        print(blkX.layout)

    thr_copy = tiled_copy.get_slice(tidx)

    if (tidx == 0 and bidx == 0):
        print(thr_copy)
    tXgX = thr_copy.partition_S(blkX)  
    tXcX = thr_copy.partition_S(cblkX) 
    tXgY = thr_copy.partition_D(blkY) # partition_S is for partition source, _D is for destination

    if (tidx == 0 and bidx == 0):
        print(tXgX.layout)


    # deletes trailing degenerate modes from partitioning
    tXgX = cute.make_tensor(tXgX.iterator, cute.get(tXgX.layout, mode=[0]))  
    tXcX = cute.make_tensor(tXcX.iterator, cute.get(tXcX.layout, mode=[0]))
    tXgY = cute.make_tensor(tXgY.iterator, cute.get(tXgY.layout, mode=[0]))
    

    



    # initialize SMEM buffers
    if widx == 0:
        reduction_buffer[lidx] = 0.0
        max_buffer[lidx] = -Float32.inf
    cute.arch.sync_threads()

    
    M, N = shape

    if (tidx == 0 and bidx == 0):
        print(tXgX.layout)
    rPred_layout = cute.make_layout((vec_size, n_iters), stride=(0,1)) 
    rPred = cute.make_fragment(rPred_layout, cutlass.Boolean) 
    if (tidx == 0 and bidx == 0):
        print(tXcX.shape)
        print("rPredveclayout")
        print(rPred.layout)

    

    tXrX = cute.make_rmem_tensor_like(tXgX)

    for ni in range(0, n_iters):
        rPred[(0,ni)] = cute.elem_less(tXcX[(vec_size - 1,ni)], shape) # if the end of a vector is oob then the whole vector is ineligible
        
        if cutlass.const_expr(tiler[1] > N): # filling -inf only necessary when tiling goes past N
            if ni == n_iters - 1:
                for i in cutlass.range_constexpr(vec_size):
                    tXrX[i, ni] = -Float32.inf

    # Loading elements from GMEM to register tensors
    cute.copy(tiled_copy, tXgX, tXrX, pred=rPred)
    # if cutlass.const_expr(N % vec_size != 0): # loading non vectorized elements for tail case
    #     for i in cutlass.range_constexpr(vec_size):
    #         if cute.elem_less(tXcX[(i, n_iters - 1)], shape):
    #             tXrX[(i, n_iters - 1)] = tXgX[(i, n_iters - 1)]

    # single pass online safe softmax
    accum = 0.0
    thread_max = -Float32.inf
    for ni in range(0, n_iters):
        # load and do the reduction
        for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
            if tidx == 255 and bidx == 15: 
                cute.printf("i=%d ni=%d tXrX=%f pred=%d tXcX=(%d, %d) shape=(%d, %d)", i, ni, tXrX[i, ni], rPred[i, ni], tXcX[i, ni][0], tXcX[i, ni][1], shape[0], shape[1])
            
            prev_max = thread_max
            thread_max = cute.arch.fmax(tXrX[i, ni], thread_max)
            accum = accum * cute.math.exp(prev_max - thread_max) + cute.math.exp(tXrX[i, ni] - thread_max)
        
        
    

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
        for i in cutlass.range_constexpr(vec_size):
            if rPred[i, ni]:
                tXgY[i, ni] = cute.math.exp(tXrX[i, ni] - full_max) / divisor
        # if cutlass.const_expr(N % vec_size == 0):
        #     for i in cutlass.range_constexpr(vec_size):
        #         if rPred[i, ni]:
        #             tXgY[i, ni] = cute.math.exp(tXrX[i, ni] - full_max) / divisor

def softmax_fwd_builder(mX, mY):
    # mX is the input tensor, mY is the output
    bdim = BLOCK_SIZE
    shape = mX.shape
    M, N = shape

    max_vec_size = 128 // mX.element_type.width
    vec_size = gcd(N, max_vec_size) # gcd is necessary because torch tensor dims arent padded for byte alignment
    bits_per_vec = vec_size * mX.element_type.width
    # vec_size = bits_per_vec // mX.element_type.width
    n_iters = (N + bdim * vec_size - 1) // (bdim * vec_size)
    print(n_iters)

    # TODO: implement async copies, backward pass, benchmark, add assertions for contiguity, 
    #       verify ncu profile, add auto reshaping incase of mulitple batch dims, and add epilogue.

    @cute.jit
    def softmax_fwd_launcher(mX, mY):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=bits_per_vec
        )

        thr_layout = cute.make_layout((bdim,), stride=((vec_size,)))
        val_layout = cute.make_layout((n_iters * vec_size), stride=(1))
        print(f"n_iters: {n_iters}")
        print(f"thr_layout: {thr_layout}")
        print(f"val_layout: {val_layout}")
        # use cute.recast_layout to make this work for arbitrary dtypes
        # tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (1, cute.size(thr_layout) * cute.size(val_layout))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
        
        print(f"tv_layout: {tv_layout}")
        print(f"tiler: {tiler}")
        cX = cute.make_identity_tensor(mX.shape) # cute.zipped_divide(cute.make_identity_tensor(mX.shape), tiler=tiler)

        fwd_kernel = softmax_fwd_kernel(mX, mY, cX, tv_layout, tiler, shape, tiled_copy, n_iters, vec_size)
        fwd_kernel.launch(
            grid=(cute.size(mX.shape[:-1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1)
        )
    return cute.compile(softmax_fwd_launcher, mX, mY)


@cute.kernel
def softmax_bwd_kernel(gY: cute.Tensor, gdY: cute.Tensor, gdX: cute.Tensor):
    
    pass


@cute.jit
def softmax_bwd_launcher(T):
    pass


if __name__ == "__main__":
    X = torch.rand(2**6,2**13+255, device='cuda', dtype=torch.float16)
    Y = torch.rand(2**6,2**13+255, device='cuda', dtype=torch.float16)

    x_ = from_dlpack(X, assumed_align=16)
    y_ = from_dlpack(Y, assumed_align=16)


    torch_softmax = torch.softmax(X.double(), dim=1).float()
    softmax_fwd_ = softmax_fwd_builder(x_, y_)
    softmax_fwd_(x_, y_)
    torch.testing.assert_close(torch_softmax, Y)
    print("success")