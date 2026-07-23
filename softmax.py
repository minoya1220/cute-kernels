import cutlass
import operator
import torch
import cutlass.cute as cute
from cutlass import Float32, Boolean, const_expr
from cutlass.cute.runtime import from_dlpack
from math import gcd

# if register pressure is concerning, consider packing bf16s
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
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    M, N = shape
    
    smem = cutlass.utils.SmemAllocator()
    buffer_layout = cute.make_layout(cute.arch.WARP_SIZE)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    # if (tidx == 0 and bidx == 0):
    #     print(cute.zipped_divide(gX, tiler).layout)

    blk_coord = (bidx, None)
    blkX = cute.local_tile(gX, tiler, blk_coord) 
    blkY = cute.local_tile(gY, tiler, blk_coord)
    cblkX = cute.local_tile(cX, tiler, blk_coord)
    
    
    # if (tidx == 0 and bidx == 0):
    #     print(blkX.layout)

    thr_copy = tiled_copy.get_slice(tidx)

    # if (tidx == 0 and bidx == 0):
    #     print(thr_copy)
    tXgX = thr_copy.partition_S(blkX)  
    tXcX = thr_copy.partition_S(cblkX) 
    tXgY = thr_copy.partition_D(blkY) # _S is for partition source, _D is for destination

    # if (tidx == 0 and bidx == 0):
    #     print(tXgX.layout)


    # deletes trailing degenerate modes from partitioning, required for later indexing to work
    tXgX = cute.make_tensor(tXgX.iterator, cute.get(tXgX.layout, mode=[0]))  
    tXcX = cute.make_tensor(tXcX.iterator, cute.get(tXcX.layout, mode=[0]))
    tXgY = cute.make_tensor(tXgY.iterator, cute.get(tXgY.layout, mode=[0]))
    

    if widx == 0:
        reduction_buffer[lidx] = 0.0
        max_buffer[lidx] = -Float32.inf
    cute.arch.sync_threads()
    

    # if (tidx == 0 and bidx == 0):
    #     print(tXgX.layout)
    rPred_layout = cute.make_layout((vec_size, n_iters), stride=(0,1)) 
    rPred = cute.make_fragment(rPred_layout, cutlass.Boolean) 
    # if (tidx == 0 and bidx == 0):
    #     print(tXcX.shape)
    #     print("rPredveclayout")
    #     print(rPred.layout)

    tXrX = cute.make_rmem_tensor_like(tXgX)

    for ni in cutlass.range_constexpr(0, n_iters): # n_iters should be bounded
        rPred[(0,ni)] = cute.elem_less(tXcX[(vec_size - 1,ni)], shape) # if the end of a vector is oob then the whole vector is ineligible
        
        if cutlass.const_expr(tiler[1] > N): # filling -inf only necessary when tiling goes past N
            if ni == n_iters - 1:
                for i in cutlass.range_constexpr(vec_size):
                    tXrX[i, ni] = tXrX.element_type(-Float32.inf)

    # Loading elements from GMEM to register tensors
    cute.copy(tiled_copy, tXgX, tXrX, pred=rPred)

    thread_max = -Float32.inf
    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
            thread_max = cute.arch.fmax(tXrX[i, ni].to(Float32), thread_max)

    # single pass online safe softmax
    accum = 0.0
    if thread_max != -Float32.inf: # if it is -inf accum is guaranteed to be 0
        for ni in cutlass.range_constexpr(0, n_iters):
            # load and do the reduction
            for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
                # if tidx == 255 and bidx == 15: 
                #     cute.printf("i=%d ni=%d tXrX=%f pred=%d tXcX=(%d, %d) shape=(%d, %d)", i, ni, tXrX[i, ni].to(Float32), rPred[i, ni], tXcX[i, ni][0], tXcX[i, ni][1], shape[0], shape[1])
                
                accum = accum + cute.math.exp(tXrX[i, ni].to(Float32) - thread_max)
        
    

    # intra-warp reduction
    warp_max = cute.arch.warp_reduction(thread_max, cute.arch.fmax)
    if warp_max != -Float32.inf: # thread_max and warp_max would have to -inf guaranteeing accum continues to be 0 
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


    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(vec_size):
            if rPred[i, ni]:
                tXgY[i, ni] = (cute.math.exp(tXrX[i, ni].to(Float32) - full_max) / divisor).to(tXgY.element_type)


def softmax_fwd_builder(X: torch.Tensor) -> callable:
    assert X.is_contiguous()
    
    Y = torch.empty_like(X) 
    mX = from_dlpack(X.detach(), assumed_align=16)
    mY = from_dlpack(Y, assumed_align=16)

    M, N = mX.shape
    bdim = 256
    assert N > 1

    max_vec_size = 128 // mX.element_type.width # 128 bits is largest ld/st instruction
    vec_size = gcd(N, max_vec_size) # gcd is necessary because torch tensor dims arent padded for byte alignment
    bits_per_vec = vec_size * mX.element_type.width
    n_iters = (N + bdim * vec_size - 1) // (bdim * vec_size)

    # TODO: benchmark, implement backward pass, 
    #       verify ncu profile, turn into pytorch op, implement async SMEM transfers, and consider adding epilogue (potentially overscope)

    @cute.jit
    def softmax_fwd_launcher(mX: cute.Tensor, mY: cute.Tensor):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=bits_per_vec
        )

        thr_layout = cute.make_layout((bdim,), stride=((vec_size,)))
        val_layout = cute.make_layout((n_iters * vec_size), stride=(1))
        # print(f"n_iters: {n_iters}")
        # print(f"thr_layout: {thr_layout}")
        # print(f"val_layout: {val_layout}")

        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (1, cute.size(thr_layout) * cute.size(val_layout))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
        
        # print(f"tv_layout: {tv_layout}")
        # print(f"tiler: {tiler}")
        cX = cute.make_identity_tensor(mX.shape) 

        fwd_kernel = softmax_fwd_kernel(mX, mY, cX, tv_layout, tiler, mX.shape, tiled_copy, n_iters, vec_size)
        fwd_kernel.launch(
            grid=(cute.size(mX.shape[:-1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1)
        )


    compiled_kernel = cute.compile(softmax_fwd_launcher, mX, mY)
    
    def kernel_wrapper(X: torch.Tensor) -> torch.Tensor:
        original_shape = X.shape
        X = X.flatten(0,-2)
        Y = torch.empty_like(X)
        compiled_kernel(X, Y)
        return Y.view(original_shape)
    
    return kernel_wrapper


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
                ):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()
    M, N = shape

    smem = cutlass.utils.SmemAllocator()
    buffer_layout = cute.make_layout(cute.arch.WARP_SIZE)
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)

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


    if widx == 0:
        reduction_buffer[lidx] = 0.0

    rPred_layout = cute.make_layout((vec_size, n_iters), stride=(0,1))
    rPred = cute.make_fragment(rPred_layout, cutlass.Boolean)

    tYrY = cute.make_rmem_tensor_like(tYgY)
    tYrdY = cute.make_rmem_tensor_like(tYgdY)


    for ni in cutlass.range_constexpr(0, n_iters):
        rPred[(0, ni)] = cute.elem_less(tYcY[(vec_size - 1, ni)], shape)

        if cutlass.const_expr(tiler[1] > N): # filling -inf only necessary when tiling goes past N
            if ni == n_iters - 1:
                for i in cutlass.range_constexpr(vec_size):
                    tYrY[i, ni] = tYrY.element_type(0.0)
                    tYrdY[i, ni] = tYrdY.element_type(0.0)
    
    cute.copy(tiled_copy, tYgY, tYrY, pred=rPred)
    cute.copy(tiled_copy, tYgdY, tYrdY, pred=rPred)

    accum = 0.0
    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(0, vec_size):
            # if tidx == 0 and bidx == 22: 
            #     cute.printf("i=%d ni=%d tYrY=%f tYrdY= %f pred=%d tYcY=(%d, %d) shape=(%d, %d)", i, ni, tYrY[i, ni].to(Float32), tYrdY[i, ni].to(Float32), rPred[i, ni], tYcY[i, ni][0], tYcY[i, ni][1], shape[0], shape[1])
            accum += tYrY[(i, ni)].to(Float32) * tYrdY[(i, ni)].to(Float32)
    

    accum = cute.arch.warp_reduction(accum, operator.add)

    cute.arch.sync_threads()
    if lidx == 0:
        reduction_buffer[widx] = accum

    cute.arch.sync_threads()
    partial_sum = reduction_buffer[lidx]
    full_sum = cute.arch.warp_reduction(partial_sum, operator.add)

    for ni in cutlass.range_constexpr(0, n_iters):
        for i in cutlass.range_constexpr(vec_size):
            if rPred[i, ni]:
                tYgdX[i, ni] = (tYrY[(i, ni)].to(Float32) * (tYrdY[(i, ni)].to(Float32) - full_sum)).to(tYgdX.element_type)
    


def softmax_bwd_builder(Y: torch.Tensor, dY: torch.Tensor) -> callable:
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


    @cute.jit
    def softmax_bwd_launcher(mY: cute.Tensor, mdY: cute.Tensor, mdX: cute.Tensor):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=bits_per_vec
        )

        thr_layout = cute.make_layout((bdim,), stride=(vec_size,))
        val_layout = cute.make_layout((n_iters * vec_size), stride=(1))

        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (1, cute.size(thr_layout) * cute.size(val_layout))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)

        cY = cute.make_identity_tensor(mY.shape)

        compiled_kernel = softmax_bwd_kernel(mY, mdY, mdX, cY, tv_layout, tiler, mY.shape, tiled_copy, n_iters, vec_size)
        compiled_kernel.launch(
            grid=(cute.size(mY.shape[:-1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1)
        )
    
    compiled_kernel = cute.compile(softmax_bwd_launcher, mY, mdY, mdX)

    def kernel_wrapper(Y: torch.Tensor, dY: torch.Tensor) -> torch.Tensor:
        original_shape = Y.shape
        Y = Y.flatten(0,-2).detach()
        dX = torch.empty_like(Y)
        compiled_kernel(Y, dY, dX)
        return dX.view(original_shape)
    
    return kernel_wrapper
        



def test_fwd(dims: list[int]):
    for dim in dims:
        X = torch.randn(2**6, dim, device='cuda', dtype=torch.bfloat16)

        torch_softmax = torch.softmax(X.double(), dim=1).to(X.dtype)
        softmax_fwd = softmax_fwd_builder(X)
        torch.testing.assert_close(torch_softmax, softmax_fwd(X))
 
def benchmark_fwd(dims: list[int], dtype=torch.bfloat16) -> list[float]:
    times = []
    
    for dim in dims:
        X = torch.randn(2**6, dim, device='cuda', dtype=dtype)
        softmax_fwd = softmax_fwd_builder(X)

        times.append(benchmark(softmax_fwd, [X]))
    
    return times

def test_bwd(dims: list[int], dtype=torch.bfloat16):
    for dim in dims:
        X = torch.randn(2**6, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(X)
        
        Y = torch.softmax(X, dim=1).to(dtype)
        Y_double, dY_double = Y.double(), dY.double()
        ref_softmax_bwd = (Y_double * (dY_double - (Y_double * dY_double).sum(dim=1, keepdim=True))).to(dtype)

        softmax_bwd = softmax_bwd_builder(Y, dY)
        torch.testing.assert_close(ref_softmax_bwd, softmax_bwd(Y, dY))


def benchmark_bwd(dims: list[int], dtype=torch.bfloat16) -> list[float]:
    times = []
    for dim in dims:
        Y = torch.randn(2**6, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(Y)
        softmax_bwd = softmax_bwd_builder()
        times.append(benchmark(softmax_bwd, [Y, dY]))
    
    return times

def benchmark(fn: callable, inputs: list[torch.Tensor], warmup=10, iters=100, L2_clear=True):
    inputs_buffers = [[input_tensor for input_tensor in inputs]]
    
    if L2_clear:
        input_size = sum([input_tensor.nbytes for input_tensor in inputs])
        buffer_size = 256 * 2**20 # 256 MB clears l2 for all modern gpus
        n_buffers = buffer_size // input_size

        inputs_buffers = [[input_tensor.clone() for input_tensor in inputs] for _ in range(0, n_buffers)]
    
    
    for _ in range(0, warmup):
        fn(*inputs_buffers[0])
    
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(0, iters):
        fn(*inputs_buffers[i % n_buffers])
    end.record()

    avg_time = start.elapsed_time(end) / iters
    return avg_time

if __name__ == "__main__":
    torch.manual_seed(123)
    test_dims = [3, 7, 8, 100, 257, 1021, 2048, 2049, 4099, 8192, 32768, 131072]
    test_fwd(test_dims)
    test_bwd(test_dims)
    print("success")