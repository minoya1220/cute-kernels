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
def softmax_bwd_kernel(
                    gY: cute.Tensor, 
                    gdY: cute.Tensor, 
                    gdX: cute.Tensor,
                    cY: cute.Tensor,
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

    # Divide global memory tensor into 
    blk_coord = (bidx, None)
    blkY = cute.local_tile(gY, tiler, blk_coord)
    blkdY = cute.local_tile(gdY, tiler, blk_coord)
    blkcY = cute.local_tile(cY, tiler, blk_coord)
    blkdX = cute.local_tile(gdX, tiler, blk_coord)


    thr_copy = tiled_copy.get_slice(tidx)
    tYgY = thr_copy.partition_S(blkY)
    tYcY = thr_copy.partition_S(blkcY)
    tYgdY = thr_copy.partition_S(blkdY)
    tYgdX = thr_copy.partition_D(blkdX)

    # deletes trailing degenerate modes from partitioning, required for later indexing to work
    tYgY = cute.make_tensor(tYgY.iterator, cute.get(tYgY.layout, mode=[0]))  
    tYcY = cute.make_tensor(tYcY.iterator, cute.get(tYcY.layout, mode=[0]))
    tYgdY = cute.make_tensor(tYgdY.iterator, cute.get(tYgdY.layout, mode=[0]))
    tYgdX = cute.make_tensor(tYgdX.iterator, cute.get(tYgdX.layout, mode=[0]))


    rPred_layout = cute.make_layout((vec_size, n_iters), stride=(0,1))
    rPred = cute.make_fragment(rPred_layout, cutlass.Boolean)
    
    tYrY = cute.make_rmem_tensor_like(tYgY)
    tYrdY = cute.make_rmem_tensor_like(tYgdY)

    # Fill predicates and oob elements
    for ni in cutlass.range_constexpr(0, n_iters):
        rPred[(0, ni)] = cute.elem_less(tYcY[(vec_size - 1, ni)], shape)

        if cutlass.const_expr(tiler[1] > N): # filling -inf only necessary when tiling goes past N
            if ni == n_iters - 1:
                for i in cutlass.range_constexpr(vec_size):
                    tYrY[i, ni] = tYrY.element_type(0.0)
                    tYrdY[i, ni] = tYrdY.element_type(0.0)
    
    # Load GMEM -> RMEM
    cute.copy(tiled_copy, tYgY, tYrY, pred=rPred)
    cute.copy(tiled_copy, tYgdY, tYrdY, pred=rPred)

    # Intra-thread reduction
    accum = 0.0
    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(0, vec_size):
            # if tidx == 255 and bidx == 0: 
            #     cute.printf("i=%d ni=%d tYrY=%f tYrdY= %f pred=%d tYcY=(%d, %d) shape=(%d, %d)", i, ni, tYrY[i, ni].to(Float32), tYrdY[i, ni].to(Float32), rPred[i, ni], tYcY[i, ni][0], tYcY[i, ni][1], shape[0], shape[1])
            accum += tYrY[(i, ni)].to(Float32) * tYrdY[(i, ni)].to(Float32)
    
    
    # Intra-warp reduction
    partial_sum = warp_reduction_partial(accum, operator.add, min(threads_per_row, 32))


    # Inter-warp reduction
    if cutlass.const_expr(threads_per_row > 32):
        if lidx == 0:
            reduction_buffer[widx] = partial_sum
        cute.arch.sync_threads()
        
        warps_per_row = threads_per_row // cute.arch.WARP_SIZE
        thr2buffer = cute.make_layout((warps_per_row, cute.arch.WARP_SIZE, num_warps // warps_per_row), 
            stride=(1, 0, warps_per_row))
        partial_sum = reduction_buffer[thr2buffer(tidx)]

    full_sum = warp_reduction_partial(partial_sum, operator.add, max(threads_per_row // 32, 1))


    # Compute and store final result
    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(vec_size):
            if rPred[i, ni]:
                tYgdX[i, ni] = (tYrY[(i, ni)].to(Float32) * (tYrdY[(i, ni)].to(Float32) - full_sum)).to(tYgdX.element_type)
    


def softmax_bwd_builder(Y: torch.Tensor, dY: torch.Tensor) -> Callable:
    assert Y.shape == dY.shape and Y.stride() == dY.stride()
    assert Y.is_contiguous()

    dX = torch.empty_like(Y)
    mY = from_dlpack(Y.detach(), assumed_align=16)
    mdY = from_dlpack(dY, assumed_align=16)
    mdX = from_dlpack(torch.empty_like(Y), assumed_align=16)

    M, N = mY.shape
    bdim = 256
    assert N > 1

    max_vec_size = 128 // mY.element_type.width
    vec_size = gcd(N, max_vec_size)
    bits_per_vec = vec_size * mY.element_type.width
    n_iters = (N + bdim * vec_size - 1) // (bdim * vec_size)

    threads_per_row = min(bdim, 1 << (N // vec_size - 1).bit_length())
    rows_per_block = bdim // threads_per_row

    @cute.jit
    def softmax_bwd_launcher(mY: cute.Tensor, mdY: cute.Tensor, mdX: cute.Tensor, stream: cuda.CUstream = cuda.CUstream(0)):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=bits_per_vec
        )
        assert not bdim > 1024 # some consumer cards allow > 1024, but t > 1024 would require an extra interwarp reduction
        
        thr_layout = cute.make_layout((threads_per_row, rows_per_block), stride=(rows_per_block*vec_size,1))
        val_layout = cute.make_layout((vec_size, n_iters), stride=(1, vec_size))

        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (rows_per_block, cute.size(tv_layout) // rows_per_block)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)

        cY = cute.make_identity_tensor(mY.shape)

        bwd_kernel = softmax_bwd_kernel(mY, mdY, mdX, cY, tv_layout, tiler, mY.shape, tiled_copy, n_iters, vec_size, threads_per_row, bdim)
        
        num_blocks = (M + rows_per_block - 1) // rows_per_block
        bwd_kernel.launch(
            grid=(num_blocks, 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1),
            stream=stream
        )
    
    compiled_kernel = cute.compile(softmax_bwd_launcher, mY, mdY, mdX)

    def kernel_wrapper(Y: torch.Tensor, dY: torch.Tensor,*, out: torch.Tensor = None) -> torch.Tensor:
        original_shape = Y.shape
        Y = Y.flatten(0,-2).detach()
        dY = dY.flatten(0,-2).detach()
        dX = torch.empty_like(Y) if out is None else out

        s = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_kernel(Y, dY, dX, s)
        return dX.view(original_shape)
    
    return kernel_wrapper