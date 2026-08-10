import cutlass
import operator
import torch
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from collections.abc import Callable
from cutlass import Float32, Boolean
from cutlass.cute.runtime import from_dlpack
from math import gcd
from utils import warp_reduction_partial


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
                    threads_per_row: cutlass.Constexpr,
                    bdim: cutlass.Constexpr
                ):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    num_warps = bdim // cute.arch.WARP_SIZE
    M, N = shape

    smem = cutlass.utils.SmemAllocator()

    buffer_layout = cute.make_layout(num_warps)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    # Divide global memory tensor 
    blk_coord = (bidx, None)
    blkX = cute.local_tile(gX, tiler, blk_coord) 
    blkY = cute.local_tile(gY, tiler, blk_coord)
    cblkX = cute.local_tile(cX, tiler, blk_coord)


    thr_copy = tiled_copy.get_slice(tidx)
    tXgX = thr_copy.partition_S(blkX)  
    tXcX = thr_copy.partition_S(cblkX) 
    tXgY = thr_copy.partition_D(blkY) # _S is for partition source, _D is for destination

    # deletes trailing degenerate modes from partitioning, required for later indexing to work
    tXgX = cute.make_tensor(tXgX.iterator, cute.get(tXgX.layout, mode=[0]))  
    tXcX = cute.make_tensor(tXcX.iterator, cute.get(tXcX.layout, mode=[0]))
    tXgY = cute.make_tensor(tXgY.iterator, cute.get(tXgY.layout, mode=[0]))
    
    
    rPred_layout = cute.make_layout((vec_size, n_iters), stride=(0,1)) 
    rPred = cute.make_fragment(rPred_layout, cutlass.Boolean) 

    tXrX = cute.make_rmem_tensor_like(tXgX)


    # Fill predicates and oob elements
    for ni in cutlass.range_constexpr(0, n_iters): # n_iters should be bounded
        rPred[(0,ni)] = cute.elem_less(tXcX[(vec_size - 1,ni)], shape) # if the end of a vector is oob then the whole vector is ineligible
        
        if cutlass.const_expr(tiler[1] > N): # filling -inf only necessary when tiling goes past N
            if ni == n_iters - 1:
                for i in cutlass.range_constexpr(vec_size):
                    tXrX[i, ni] = tXrX.element_type(-Float32.inf)


    # Loading elements from GMEM to register tensors
    cute.copy(tiled_copy, tXgX, tXrX, pred=rPred)


    # Intra-thread max
    thread_max = -Float32.inf
    for ni in cutlass.range_constexpr(n_iters):
        for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
            thread_max = cute.arch.fmax(tXrX[i, ni].to(Float32), thread_max)

    # Intra-thread safe softmax numerator
    accum = 0.0
    if thread_max != -Float32.inf: # if it is -inf accum is guaranteed to be 0
        for ni in cutlass.range_constexpr(n_iters):
            for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
                # if tidx == 255 and bidx == 15: 
                #     cute.printf("i=%d ni=%d tXrX=%f pred=%d tXcX=(%d, %d) shape=(%d, %d)", i, ni, tXrX[i, ni].to(Float32), rPred[i, ni], tXcX[i, ni][0], tXcX[i, ni][1], shape[0], shape[1])
                
                accum = accum + cute.math.exp(tXrX[i, ni].to(Float32) - thread_max)
        

    # Intra-warp reduction
    partial_max = warp_reduction_partial(thread_max, cute.arch.fmax, min(threads_per_row, 32))
    if partial_max != -Float32.inf: # thread_max and partial_max would have to -inf guaranteeing accum continues to be 0 
        accum = accum * cute.math.exp(thread_max - partial_max) 
        accum = warp_reduction_partial(accum, operator.add, min(threads_per_row, 32))
    partial_sum = accum
    

    # Inter-warp reduction
    if cutlass.const_expr(threads_per_row > 32):
        if lidx == 0:
            reduction_buffer[widx] = partial_sum
            max_buffer[widx] = partial_max
        cute.arch.sync_threads()

        warps_per_row = threads_per_row // cute.arch.WARP_SIZE 
        
        # # naive addressing
        # row_start = widx // warps_per_row * warps_per_row
        # idx = row_start + lidx % warps_per_row
        # partial_max = max_buffer[idx] 
        # partial_sum = reduction_buffer[idx]

        # # implemented using CuTe Layouts 
        # thr2buffer = cute.make_layout( ((warps_per_row, cute.arch.WARP_SIZE // warps_per_row),(warps_per_row, num_warps // warps_per_row)),
        #                                  stride=((1, 0), (0, warps_per_row))) 
        # partial_max = max_buffer[thr2buffer((lidx, widx))] # (lidx, widx) can be replaced by tidx
        # partial_sum = reduction_buffer[thr2buffer((lidx, widx))]
        
        # thr2buffer coalesces into ((warps_per_row, cute.arch.WARP_SIZEs, num_warps // warps_per_row):(1, 0, warps_per_row)), this layout can only be indexed into using tidx 
        thr2buffer = cute.make_layout((warps_per_row, cute.arch.WARP_SIZE, num_warps // warps_per_row), 
            stride=(1, 0, warps_per_row))
        partial_max = max_buffer[thr2buffer(tidx)]
        partial_sum = reduction_buffer[thr2buffer(tidx)]
        # kept this for fun, naive addressing is clearer imo
        
    full_max = warp_reduction_partial(partial_max, cute.arch.fmax, max(threads_per_row//32, 1))
    partial_sum = partial_sum * cute.math.exp(partial_max - full_max)
    divisor = warp_reduction_partial(partial_sum, operator.add, max(threads_per_row//32, 1))


    # Compute and store result 
    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(vec_size):
            if rPred[i, ni]:
                tXgY[i, ni] = (cute.math.exp(tXrX[i, ni].to(Float32) - full_max) / divisor).to(tXgY.element_type)



def softmax_fwd_builder(X: torch.Tensor) -> Callable:
    assert X.is_contiguous()
    
    Y = torch.empty_like(X) 
    mX = from_dlpack(X.detach(), assumed_align=16)
    mY = from_dlpack(Y, assumed_align=16)

    M, N = mX.shape
    bdim = 1024 # 256, 512, 1024 all work, gives different perf for >2**16
    assert N > 1

    max_vec_size = 128 // mX.element_type.width # 128 bits is largest ld/st instruction
    vec_size = gcd(N, max_vec_size) # guarantees alignment for vector loads
    bits_per_vec = vec_size * mX.element_type.width
    n_iters = (N + bdim * vec_size - 1) // (bdim * vec_size) # (a + b - 1) // b is cdiv for positive numbers 

    threads_per_row = min(bdim, 1 << (N // vec_size - 1).bit_length()) # 1 << (k-1).bit_length() rounds up to nearest pow2
    # rounding to pow2 allows us to keep using warp shuffles to reduce
    rows_per_block = bdim // threads_per_row 


    @cute.jit
    def softmax_fwd_launcher(mX: cute.Tensor, mY: cute.Tensor, stream: cuda.CUstream = cuda.CUstream(0)):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=bits_per_vec
        )
        assert not bdim > 1024 # some consumer cards allow > 1024, but t > 1024 would require an extra interwarp reduction
        
        thr_layout = cute.make_layout((threads_per_row, rows_per_block), stride=(rows_per_block*vec_size,1))
        val_layout = cute.make_layout((vec_size, n_iters), stride=(1, vec_size))
        # print(f"thr_layout: {thr_layout}")
        # print(f"val_layout: {val_layout}")
        # print(f"n_iters: {n_iters}")
        # print(f"threads_per_row: {threads_per_row}")
        # print(f"vec_size: {vec_size}")

        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (rows_per_block, cute.size(tv_layout) // rows_per_block)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
        
        # print(f"tv_layout: {tv_layout}")
        # print(f"tiler: {tiler}")
        cX = cute.make_identity_tensor(mX.shape) 

        fwd_kernel = softmax_fwd_kernel(mX, mY, cX, tv_layout, tiler, mX.shape, tiled_copy, n_iters, vec_size, threads_per_row, bdim)
        
        num_blocks = (M + rows_per_block - 1) // rows_per_block
        fwd_kernel.launch(
            grid=(num_blocks, 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1),
            stream=stream
        )


    compiled_kernel = cute.compile(softmax_fwd_launcher, mX, mY)
    
    def kernel_wrapper(X: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
        original_shape = X.shape
        X = X.flatten(0,-2)
        Y = torch.empty_like(X) if out is None else out
        
        s = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_kernel(X, Y, stream=s)
        
        return Y.view(original_shape)
    
    return kernel_wrapper


        






# TODO: split into multiple files, explore packed b16s, do a writeup

# # rewrite to use layouts to organize reduction buffer
# warp indexing bug fix 
# widx = tidx // 32
# lidx = tidx % 32
# warps_per_row = threads_per_row // 32
# row_base = ((tidx // 32) // warps_per_row) * warps_per_row
# idx = row_base + (tidx % 32) % warps_per_row
# partial_max = max_buffer[idx]
# partial_sum = reduction_buffer[idx]

# (wpr, 32 / wpr) : (1, 0) index using [lidx] 
# (num_warps / wpr, wpr) : (0, 1)  index using [widx]
# full layout = ((wpr, num_lanes / wpr), (wpr, num_warps / wpr)) : ((1, 0), (0, wpr))
# widx 0-3
# wpr = 2
# widx 0, 1 -> idx 0, widx 2, 3 -> idx 1
# (2, 2) : (0, 1)  index using [widx]
# 

# ncu --set full -f -o softmax_fwd -k regex:cutlass_softmax_fwd --launch-skip 3 --launch-count 5 uv run softmax.py