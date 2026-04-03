# reimplementation of vector add educational notebook from NVIDIA
import torch
from functools import partial
from typing import List
from operator import mul, add
import cutlass 
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def vector_add_kernel_v1(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    # the g in gA, gB, gC identifies which memory the tensor is stored in
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bidx * bdim + tidx # global tid

    m, n = gA.shape # dimensions can be retrieved from the cute.Tensor 

    mi = tid // n # map 1D tid to 2d indices
    ni = tid % n

    gC[mi, ni] = gA[mi, ni] + gB[mi, ni] # load from global -> add -> write to global

@cute.kernel
def vector_add_kernel_v2(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bidx * bdim + tidx # global tid

    m, n = gA.shape[1] # gA has been zip divided so
    mi = tid // n
    ni = tid % n

    # doing None on a dim grabs everything, .load() moves into registers
    rA = gA[(None, (mi, ni))].load()
    rB = gB[(None, (mi, ni))].load()
    # prints at compile time
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gA = {gB[(None, (mi, ni))]}")

    gC[(None, (mi, ni))] = rA + rB

@cute.kernel
def vector_add_kernel_v3(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)

    # logical coord -> address
    blkA = gA[blk_coord] # (TileM, TileN) -> physical address
    blkB = gB[blk_coord] 
    blkC = gC[blk_coord] 

    tidfrgA = cute.composition(blkA, tv_layout)
    tidfrgB = cute.composition(blkB, tv_layout)
    tidfrgC = cute.composition(blkC, tv_layout)

    print(f"[DSL INFO] Composed with TV layout: {tidfrgA}")

    thr_coord = (tidx, None)

    thrA = tidfrgA[thr_coord]
    thrB = tidfrgB[thr_coord]
    thrC = tidfrgC[thr_coord]
    
    thrC[None] = thrA.load() + thrB.load()


@cute.kernel
def vector_add_kernel_v4(
    op: cutlass.Constexpr, # any arbitrary binary op instead of just addition
    cC: cute.Tensor, # coordinate tensor for guarding loads and stores
    gInputs: List[cute.Tensor], # holds gA and gB
    gC: cute.Tensor, 
    tv_layout: cute.Layout,
    shape: cute.Shape):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx) # used for grabbing just the portion of this thread block out ofo inputs and outputs

    blkInputs = [t[blk_coord] for t in gInputs] # can use iterable to slice across multiple inputs
    blkC = gC[blk_coord]
    blkcC = cC[blk_coord]

    frgInputs = [cute.composition(t, tv_layout) for t in blkInputs]
    frgC = cute.composition(blkC, tv_layout)
    frgcC = cute.composition(blkcC, tv_layout)

    thr_crd = (tidx, cute.repeat_like(None, frgInputs[0].shape[1])) # cute.repeat_like will just copy over whatever value layout is there


    thrInputs = [t[thr_crd] for t in frgInputs] # selects just the values for this thread
    thrC = frgC[thr_crd]
    thrcC = frgcC[thr_crd] # does the same selection in the coordinate tensor

    rPred = cute.make_fragment(thrcC.shape, cutlass.Boolean) # actually allocates registers for the predicate
    if cute.elem_less(thrcC[cute.size(thrcC) - 1], shape):
        result = op(*[t.load() for t in thrInputs]) # unpacks the input into arguments for op
        thrC.store(result)
    # else:
        # predicated path, non vectorized  ## comment this if statement if broken,
        #### BROKEN, FIX LATER #####
        # rInputs = []
        # for thrInput in thrInputs:
        #     r = cute.make_fragment_like(thrInput)
        #     cute.full_like(r, 0, dtype=r._dtype)
        #     cute.copy(thrInput, r, rPred)
        #     rInputs.append(r)
        # cute.copy(op(*rInputs), thrC, rPred)

        # any element wise op with any number of inputs now works with this kernel it just need to be passed at compile time

@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, op: cutlass.Constexpr=None):
    # not sure what the m in mA and mB refers to, could be for "matrix" but it essentially represents raw input tensor
    
    inputs = [mA, mB]
    result = mC

    m, n = mA.shape

    # # ------ V1 -------
    # block_size = 256

    # v1_kernel = vector_add_kernel_v1(mA, mB, mC)

    # v1_kernel.launch(
    #     grid=((m*n + block_size - 1) // block_size, 1, 1),
    #     block=(block_size, 1, 1)
    # )
    # # -------------------

    # ------ V2 --------
    # block_size = 256

    # gA = cute.zipped_divide(mA, tiler=(1,8)) # creating 1x4 tiles across our whole tensor for vectorization
    # gB = cute.zipped_divide(mB, tiler=(1,8)) # requires N dimension to be divisible by 4
    # gC = cute.zipped_divide(mC, tiler=(1,8)) 

    # print(f"[DSL INFO] gA = {gA}")
    # print(f"[DSL INFO] gB = {gB}")
    # print(f"[DSL INFO] gC = {gC}")

    # v2_kernel = vector_add_kernel_v2(gA, gB, gC)

    # v2_kernel.launch( # divide N by 4 now that each thread is responsible for 4 elements
    #     grid=((m * (n // 8) + block_size - 1) // block_size, 1, 1),
    #     block=(block_size, 1, 1)
    # )
    # -------------------

    # ----- V3 ------
    # coalesced_bytes = 16 # 128 bits in a vector load
    # num_rows_per_thread = 16

    # assert all(mA.element_type == t.element_type for t in [mA, mB, mC])
    # dtype = mA.element_type

    # thr_layout = cute.make_ordered_layout((4, 64), order=(1,0))
    # val_layout = cute.make_ordered_layout((num_rows_per_thread, coalesced_bytes), order=(1, 0))
    # val_layout = cute.recast_layout(dtype.width, 8, val_layout) 
    # # first output is the element-wise product of the two layouts, it define the shape of a single block
    # tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout) 
    # print(f"[DSL INFO] tiler_mn: {tiler_mn}") # tiler mn
    # print(f"[DSL INFO] TV layout: {tv_layout} ")

    # gA = cute.zipped_divide(mA, tiler=tiler_mn)
    # gB = cute.zipped_divide(mB, tiler=tiler_mn)
    # gC = cute.zipped_divide(mC, tiler=tiler_mn) 

    # vector_add_kernel_v3(gA, gB, gC, tv_layout).launch(
    #     grid=[cute.size(gC, mode=[1]), 1, 1],
    #     block=[cute.size(tv_layout, mode=[0]), 1, 1]
    # )
    # -------------------

    # ------ V4 --------
    coalesced_bytes = 16 # max size of a memory transaction (128b)

    assert all(t.element_type == inputs[0].element_type for t in inputs)
    dtype = inputs[0].element_type

    thr_layout = cute.make_ordered_layout((4,64), order=(1,0))
    val_layout = cute.make_ordered_layout((16, coalesced_bytes), order=(1,0))
    val_layout = cute.recast_layout(dtype.width, 8, val_layout) # dtype.width is in bits, divide that by 8 bits in a byte
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mInputs = [cute.zipped_divide(input, tiler_mn) for input in inputs] # partition our blocks using our tv layout
    mC = cute.zipped_divide(result, tiler_mn) # same division
    cC = cute.make_identity_tensor(result.shape)
    cC = cute.zipped_divide(cC, tiler=tiler_mn)
    remap_block = cute.make_ordered_layout(
        cute.select(mInputs[0].shape[1], mode=[1,0]), order=(1,0)
    )
    for i, t in enumerate(mInputs):
        mInputs[i] = cute.composition(t, (None, remap_block))
    
    mC = cute.composition(mC, (None, remap_block))

    
    cC = cute.composition(cC, (None, remap_block))
    vector_add_kernel_v4(op, cC, mInputs, mC, tv_layout, result.shape).launch(
        grid=[cute.size(mC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout,mode=[0]), 1, 1]
    )
    # -------------------

M, N = 1024, 1024

A = torch.randn(M,N, device='cuda', dtype=torch.bfloat16)
B = torch.randn(M,N, device='cuda', dtype=torch.bfloat16)
C = torch.randn(M,N, device='cuda', dtype=torch.bfloat16)

total_elem = M * N * 3

# converts torch tensors to CuTe tensors
a_ = from_dlpack(A, assumed_align=16)
b_ = from_dlpack(B, assumed_align=16)
c_ = from_dlpack(C, assumed_align=16)

op = add
def swiglu(a, b):
    return a * b * (1.0 / (1.0 + cute.exp(-b)))
vector_add_ = cute.compile(vector_add, a_, b_, c_, op)

vector_add_(a_, b_, c_)

# torch.testing.assert_close(C, A + B)
print("test run successful")

def benchmark(func: callable, a_: cute.Tensor, b_ : cute.Tensor, c_: cute.Tensor, op=None):
    # cute dsl has built-in benchmarking tooling
    # output is time in µs
    avg_time = cute.testing.benchmark(
        func,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )
    
    
    total_size = total_elem * 2 # 2 bytes per element
    bandwidth_usage = total_size / (avg_time) / 1000 # div by 10^3 to convert from MB/s to GB/s
    print(f"Average time: {avg_time} µs")
    print(f"Bandwidth Usage: {bandwidth_usage} GB/s")


benchmark(vector_add_, a_, b_, c_)
