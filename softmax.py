import cutlass
import operator
import torch
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from collections.abc import Iterable, Callable
from cutlass import Float32, Boolean, const_expr
from cutlass.cute.runtime import from_dlpack
from math import gcd

def warp_reduction_partial(value: cute.typing.Numeric, op: Callable, width: int) -> cute.typing.Numeric:
    assert 1 <= width and width <= 32, f"width must be in [1,32], got {width}"
    assert (width & (width - 1)) == 0, f"width must be a power of 2, got {width}"
    result = value
    for i in range(width.bit_length() - 1):
        result = op(result, cute.arch.shuffle_sync_bfly(result,1 << i))
    return result

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
                    threads_per_row: cutlass.Constexpr,
                    bdim: cutlass.Constexpr
    ):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    widx, lidx = cute.arch.warp_idx(), cute.arch.lane_idx()

    M, N = shape

    smem = cutlass.utils.SmemAllocator()

    num_warps = bdim // cute.arch.WARP_SIZE
    buffer_layout = cute.make_layout(num_warps)

    reduction_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    max_buffer = smem.allocate_tensor(cutlass.Float32, buffer_layout, byte_alignment=16)
    
    # if (tidx == 0 and bidx == 0):
    #     print(cute.zipped_divide(gX, tiler).layout)

    blk_coord = (bidx, None)
    blkX = cute.local_tile(gX, tiler, blk_coord) 
    blkY = cute.local_tile(gY, tiler, blk_coord)
    cblkX = cute.local_tile(cX, tiler, blk_coord)
    
    
    
    # if (tidx == 0 and bidx == 0):
    #     print(f"blkX: {blkX.layout}")

    thr_copy = tiled_copy.get_slice(tidx)

    # if (tidx == 127 and bidx == 0):
    #     print(thr_copy)
    tXgX = thr_copy.partition_S(blkX)  
    tXcX = thr_copy.partition_S(cblkX) 
    tXgY = thr_copy.partition_D(blkY) # _S is for partition source, _D is for destination

    # if (tidx == 0 and bidx == 0):
    #     print(f"tXgX: {tXgX.layout}")


    # deletes trailing degenerate modes from partitioning, required for later indexing to work
    tXgX = cute.make_tensor(tXgX.iterator, cute.get(tXgX.layout, mode=[0]))  
    tXcX = cute.make_tensor(tXcX.iterator, cute.get(tXcX.layout, mode=[0]))
    tXgY = cute.make_tensor(tXgY.iterator, cute.get(tXgY.layout, mode=[0]))
    
    # if (tidx == 0 and bidx == 0):
    #     print(f"tXgX (trailing modes cleared): {tXgX.layout}")


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

    # intra-thread max
    thread_max = -Float32.inf
    for ni in cutlass.range_constexpr(n_iters):
        for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
            thread_max = cute.arch.fmax(tXrX[i, ni].to(Float32), thread_max)

    # intra-thread safe softmax numerator
    accum = 0.0
    if thread_max != -Float32.inf: # if it is -inf accum is guaranteed to be 0
        for ni in cutlass.range_constexpr(n_iters):
            for i in cutlass.range_constexpr(vec_size): # range constexpr loops get unrolled
                # if tidx == 255 and bidx == 15: 
                #     cute.printf("i=%d ni=%d tXrX=%f pred=%d tXcX=(%d, %d) shape=(%d, %d)", i, ni, tXrX[i, ni].to(Float32), rPred[i, ni], tXcX[i, ni][0], tXcX[i, ni][1], shape[0], shape[1])
                
                accum = accum + cute.math.exp(tXrX[i, ni].to(Float32) - thread_max)
        
    

    # intra-warp reduction
    warp_max = warp_reduction_partial(thread_max, cute.arch.fmax, min(threads_per_row, 32))

    if warp_max != -Float32.inf: # thread_max and warp_max would have to -inf guaranteeing accum continues to be 0 
        accum = accum * cute.math.exp(thread_max - warp_max) 
        accum = warp_reduction_partial(accum, operator.add, min(threads_per_row, 32))
    
    partial_result = accum
    partial_max = warp_max
    
    # inter-warp reduction
    if cutlass.const_expr(threads_per_row > 32):
        # write each warp result to SMEM
        if lidx == 0:
            reduction_buffer[widx] = accum
            max_buffer[widx] = warp_max

        cute.arch.sync_threads()

        warps_per_row = threads_per_row // cute.arch.WARP_SIZE 
        
        # # naive addressing
        # row_start = widx // warps_per_row * warps_per_row
        # idx = row_start + lidx % warps_per_row
        # partial_max = max_buffer[idx] 
        # partial_result = reduction_buffer[idx]

        # # implemented using CuTe Layouts 
        # thr2buffer = cute.make_layout( ((warps_per_row, cute.arch.WARP_SIZE // warps_per_row),(warps_per_row, num_warps // warps_per_row)),
        #                                  stride=((1, 0), (0, warps_per_row))) 
        # partial_max = max_buffer[thr2buffer((lidx, widx))] # (lidx, widx) can be replaced by tidx
        # partial_result = reduction_buffer[thr2buffer((lidx, widx))]
        
        # thr2buffer coalesces into ((warps_per_row, cute.arch.WARP_SIZEs, num_warps // warps_per_row):(1, 0, warps_per_row)), this layout can only be indexed into using tidx 
        thr2buffer = cute.make_layout((warps_per_row, cute.arch.WARP_SIZE, num_warps // warps_per_row), stride=(1, 0, warps_per_row))
        partial_max = max_buffer[thr2buffer(tidx)]
        partial_result = reduction_buffer[thr2buffer(tidx)]
        
        # kept this for fun, naive addressing is clearer imo
        

    full_max = warp_reduction_partial(partial_max, cute.arch.fmax, max(threads_per_row//32, 1))
    partial_result = partial_result * cute.math.exp(partial_max - full_max)
    divisor = warp_reduction_partial(partial_result, operator.add, max(threads_per_row//32, 1))

    # compute and store result 
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

    threads_per_row = min(bdim,1 << (N // vec_size - 1).bit_length()) # 1 << (k-1).bit_length() rounds up to nearest pow2
    # rounding to pow2 allows us to keep using warp shuffles to reduce
    rows_per_block = bdim // threads_per_row 


    @cute.jit
    def softmax_fwd_launcher(mX: cute.Tensor, mY: cute.Tensor, stream: cuda.CUstream = cuda.CUstream(0)):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=bits_per_vec
        )

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
        num_blocks = (cute.size(mX.shape[0]) + rows_per_block - 1) // rows_per_block
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


    @cute.jit
    def softmax_bwd_launcher(mY: cute.Tensor, mdY: cute.Tensor, mdX: cute.Tensor, stream: cuda.CUstream = cuda.CUstream(0)):
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mY.element_type,
            num_bits_per_copy=bits_per_vec
        )
        assert not bdim > 1024 # some consumer cards allow > 1024, but t > 1024 would require an extra interwarp reduction
        thr_layout = cute.make_layout((bdim,), stride=(vec_size,))
        val_layout = cute.make_layout((n_iters * vec_size), stride=(1))

        tv_layout = cute.logical_product(thr_layout, val_layout)
        tiler = (1, cute.size(thr_layout) * cute.size(val_layout))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)

        cY = cute.make_identity_tensor(mY.shape)

        compiled_kernel = softmax_bwd_kernel(mY, mdY, mdX, cY, tv_layout, tiler, mY.shape, tiled_copy, n_iters, vec_size)
        compiled_kernel.launch(
            grid=(cute.size(mY.shape[:-1]), 1, 1),
            block=(cute.size(tv_layout, mode=[0]), 1, 1),
            stream=stream
        )
    
    compiled_kernel = cute.compile(softmax_bwd_launcher, mY, mdY, mdX)

    def kernel_wrapper(Y: torch.Tensor, dY: torch.Tensor,*, out: torch.Tensor = None) -> torch.Tensor:
        original_shape = Y.shape
        Y = Y.flatten(0,-2).detach()
        dX = torch.empty_like(Y) if out is None else out

        s = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled_kernel(Y, dY, dX, s)
        return dX.view(original_shape)
    
    return kernel_wrapper
        
BENCH_SIZE = 2**29

def test_fwd(dims: list[int], dtype=torch.bfloat16):
    for dim in dims:
        X = torch.randn(2**6, dim, device='cuda', dtype=dtype)

        torch_softmax = torch.softmax(X.double(), dim=1).to(X.dtype)
        softmax_fwd = softmax_fwd_builder(X)
        try:
            torch.testing.assert_close(torch_softmax, softmax_fwd(X))
            print(f"dim={dim} passed")

        except AssertionError as e:
            print(f"dim={dim} failed")

def benchmark_fwd(dims: list[int], dtype=torch.bfloat16) -> list[float]:
    times = []
    
    for dim in dims:
        p = torch.cuda.get_device_properties(0)
        M = (BENCH_SIZE // dim + p.multi_processor_count - 1) // p.multi_processor_count * p.multi_processor_count
        M =  BENCH_SIZE // dim
        X = torch.randn(M, dim, device='cuda', dtype=dtype)
        Y = torch.empty_like(X)
        softmax_fwd = softmax_fwd_builder(X)

        times.append(benchmark(softmax_fwd, [X], [Y]))
    
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
        Y = torch.randn(BENCH_SIZE//dim, dim, device='cuda', dtype=dtype)
        dY = torch.randn_like(Y)
        dX = torch.empty_like(Y)
        softmax_bwd = softmax_bwd_builder(Y, dY)
        times.append(benchmark(softmax_bwd, [Y, dY], [dX]))
    
    return times

def benchmark(fn: Callable, inputs: Iterable[torch.Tensor] | torch.Tensor = (), outputs: Iterable[torch.Tensor] | torch.Tensor = (),*, warmup=20, iters=100, L2_clear=True):
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    outputs = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)

    inputs_buffers = [inputs]
    outputs_buffers = [outputs]
    n_buffers = 1

    if L2_clear:
        assert inputs and outputs, "Input and Output Tensors are required to rotate L2"
        size = sum([t.nbytes for t in (*inputs, *outputs)])
        buffer_size = 256 * 2**20 # 256 MB clears l2 for all modern gpus
        n_buffers =  min(iters, buffer_size // size + 1)

        inputs_buffers = [tuple(t.clone() for t in inputs) for _ in range(0, n_buffers)]
        outputs_buffers = [tuple(t.clone() for t in outputs) for _ in range(0, n_buffers)]

    num_outs = len(outputs_buffers[0])
    if num_outs == 0:
        caller = lambda i, o: fn(*i)
    elif num_outs == 1:
        caller = lambda i, o: fn(*i, out=o[0]) 
    else:
        caller = lambda i, o: fn(*i, out=o)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(0, warmup):
            caller(inputs_buffers[i % n_buffers], outputs_buffers[i % n_buffers])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(0, iters):
            caller(inputs_buffers[i % n_buffers], outputs_buffers[i % n_buffers])
    
    torch.cuda.synchronize()
    times = []
    for i in range(5):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    times.sort()
    median_time = times[len(times) // 2]
    return median_time


import threading, time, pynvml

class MeasureMemoryClock():
    def __init__(self, interval=0.05):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []
        self._stop = threading.Event()
        self.interval = interval
        
    def sample_clk(self, interval):
        while not self._stop.is_set():
            self.samples.append(
                pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            )
            self._stop.wait(interval)
    
    def __enter__(self):
        self.thread = threading.Thread(target=self.sample_clk, args=(self.interval,), daemon=True)
        self.thread.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self.thread.join()
        self.samples.sort()
        self.median = self.samples[len(self.samples) // 2]

if __name__ == "__main__":
    
    torch.manual_seed(123)

    test_dims = [3, 7, 8, 64, 100, 257, 1021, 2048, 2049, 4099, 8192, 32768, 131072]
    bench_dims = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    # test_fwd(bench_dims + test_dims)
    # test_bwd(test_dims)

    with MeasureMemoryClock(interval=0.01) as mem_clk:
        fwd_times = benchmark_fwd(bench_dims)
    # bwd_times = benchmark_bwd(bench_dims)
    



    def plot_fwd(dims, times, dtype=torch.bfloat16, *, elems=BENCH_SIZE, sizes=None, font='Lora', path='fwd_bench.png', memory_clock=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        c = dict(bg='#1d2021', fg='#ebdbb2', muted='#928374', grid='#32302f',
                spine='#504945', band='#282828', line='#83a598')
        s = {**dict(base=11.5, title=13.5, axis_label=11.5, tick=9.5, ref=9.5),
            **(sizes or {})}

        p = torch.cuda.get_device_properties(0)
        memory_clock = p.memory_clock_rate * 1e-3 if memory_clock is None else memory_clock
        peak = p.memory_bus_width / 8 * memory_clock * 1e6 * 2 / 1e9

        # empirical 1r+1w reference: a plain torch copy of the same total size
        n = elems * 2 // (torch.finfo(dtype).bits // 8)
        src = torch.randn(n, device='cuda', dtype=dtype)
        dst = torch.empty_like(src)
        t_ref = benchmark(lambda a, *, out: out.copy_(a), [src], [dst])
        ref = 2 * src.nbytes / (t_ref * 1e-3) / 1e9

        esize = torch.finfo(dtype).bits // 8
        gb = [2 * elems * esize / (t * 1e-3) / 1e9 for t in times]

        rc = {
            'font.family': 'serif',
            'font.serif': [font, 'TeX Gyre Pagella', 'DejaVu Serif'],
            'mathtext.fontset': 'cm', 'font.size': s['base'],
            'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
            'savefig.facecolor': c['bg'],
            'text.color': c['fg'], 'axes.labelcolor': c['fg'],
            'xtick.color': c['muted'], 'ytick.color': c['muted'],
            'axes.edgecolor': c['spine'], 'axes.linewidth': 0.8,
            'xtick.major.size': 3, 'ytick.major.size': 3,
            'axes.spines.top': False, 'axes.spines.right': False,
        }

        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(7.6, 4.8),
                                constrained_layout=dict(w_pad=0.18, h_pad=0.16))
            top, xr = peak * 1.10, dims[-1] * 1.42

            ax.set_axisbelow(True)
            ax.grid(axis='y', color=c['grid'], lw=0.7)
            ax.axhspan(peak, top, color=c['band'], lw=0, zorder=0)
            ax.axhline(ref, ls='--', lw=1.0, color=c['muted'], zorder=2)
            ax.axhline(peak, ls=':', lw=1.0, color=c['muted'], zorder=2)

            ax.fill_between(dims, gb, color=c['line'], alpha=0.10, zorder=1)
            ax.plot(dims, gb, 'o-', ms=5, lw=2, color=c['line'], zorder=3)

            # sit just above each line, left-inset from the spine
            for y, txt in [(peak, f'theoretical peak · {peak:.0f} GB/s'),
                        (ref, f'torch copy reference (1r+1w) · {ref:.0f} GB/s')]:
                ax.annotate(txt, xy=(0.015, y), xycoords=ax.get_yaxis_transform(),
                            xytext=(0, 3), textcoords='offset points',
                            fontsize=s['ref'], color=c['muted'],
                            ha='left', va='bottom', zorder=4)

            ax.set_xscale('log', base=2)
            ax.set_xticks(dims)
            ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right',
                            rotation_mode='anchor', fontsize=s['tick'])
            ax.tick_params(axis='y', labelsize=s['tick'])
            ax.set_xlim(dims[0] / 1.55, xr)
            ax.set_ylim(0, top)
            ax.set_xlabel('row length $N$', labelpad=8, fontsize=s['axis_label'])
            ax.set_ylabel('achieved bandwidth (GB/s)', labelpad=8,
                        fontsize=s['axis_label'])
            ax.set_title('Softmax Forward', fontsize=s['title'], loc='left',
                        color=c['line'], pad=12)

            sec = ax.secondary_yaxis(
                'right', functions=(lambda g: g / peak * 100, lambda q: q / 100 * peak))
            sec.set_ylabel('% of theoretical peak', labelpad=10,
                        fontsize=s['axis_label'])
            sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
            sec.spines['right'].set_visible(True)
            sec.spines['right'].set_color(c['spine'])
            sec.tick_params(colors=c['muted'], labelsize=s['tick'])
            sec.yaxis.label.set_color(c['fg'])

            fig.savefig(path, dpi=150)
            plt.close(fig)

    plot_fwd(bench_dims, fwd_times)
    # plot_bench(bench_dims, bwd_times, 3, 'bwd')  
    
    # from torch.profiler import profile, ProfilerActivity
    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=True,
    # ) as prof:
    #     benchmark_fwd([2**14])
    # prof.export_chrome_trace("trace.json")

    # benchmark_fwd([2**15])

    
    print("success")

# TODO: generate plots based on locked clocks, port multiple rows per block to backwards, split into multiple files, explore packed b16s, do a writeup

# # rewrite to use layouts to organize reduction buffer
# warp indexing bug fix 
# widx = tidx // 32
# lidx = tidx % 32
# warps_per_row = threads_per_row // 32
# row_base = ((tidx // 32) // warps_per_row) * warps_per_row
# idx = row_base + (tidx % 32) % warps_per_row
# partial_max = max_buffer[idx]
# partial_result = reduction_buffer[idx]

# (wpr, 32 / wpr) : (1, 0) index using [lidx] 
# (num_warps / wpr, wpr) : (0, 1)  index using [widx]
# full layout = ((wpr, num_lanes / wpr), (wpr, num_warps / wpr)) : ((1, 0), (0, wpr))
# widx 0-3
# wpr = 2
# widx 0, 1 -> idx 0, widx 2, 3 -> idx 1
# (2, 2) : (0, 1)  index using [widx]
# 

# ncu --set full -f -o softmax_fwd -k regex:cutlass_softmax_fwd --launch-skip 3 --launch-count 1 uv run softmax.py