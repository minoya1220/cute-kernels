# using vector add educational notebook from NVIDIA
import torch
from functools import partial
from typing import List

import cutlass 
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def vector_add_kernel_noob(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
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
def vector_add_kernel_pro(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
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
def vector_add_kernel_hacker(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout):
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
def vector_add_kernel_god(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor, tv_layout: cute.Layout):
    pass


@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # not sure what the m in mA and mB refers to, could be for "matrix" but it essentially represents raw input tensor
    

    m, n = mA.shape

    # # ------ NOOB -------
    # block_size = 256

    # noob_kernel = vector_add_kernel_noob(mA, mB, mC)

    # noob_kernel.launch(
    #     grid=((m*n + block_size - 1) // block_size, 1, 1),
    #     block=(block_size, 1, 1)
    # )
    # # -------------------

    # ------ PRO --------
    block_size = 256

    gA = cute.zipped_divide(mA, tiler=(1,8)) # creating 1x4 tiles across our whole tensor for vectorization
    gB = cute.zipped_divide(mB, tiler=(1,8)) # requires N dimension to be divisible by 4
    gC = cute.zipped_divide(mC, tiler=(1,8)) 

    print(f"[DSL INFO] gA = {gA}")
    print(f"[DSL INFO] gB = {gB}")
    print(f"[DSL INFO] gC = {gC}")

    pro_kernel = vector_add_kernel_pro(gA, gB, gC)

    pro_kernel.launch( # divide N by 4 now that each thread is responsible for 4 elements
        grid=((m * (n // 8) + block_size - 1) // block_size, 1, 1),
        block=(block_size, 1, 1)
    )
    # -------------------

    # ----- HACKER ------

    # -------------------

    # ------ GOD --------

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

vector_add_ = cute.compile(vector_add, a_, b_, c_)

vector_add_(a_, b_, c_)

torch.testing.assert_close(C, A + B)
print("test run successful")

def benchmark(func: callable, a_: cute.Tensor, b_ : cute.Tensor, c_: cute.Tensor):
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



# print("file launched")
# import cutlass
# import cutlass.cute as cute
# print("imported")

# @cute.kernel
# def kernel():
#     tid, _, _ = cute.arch.thread_idx()
#     if tid == 0:
#         cute.printf("dwadowjad")

# @cute.jit
# def kernel_launcher():
#     cute.printf("launching kernel...")

#     kernel().launch(
#         grid=(1,1,1),
#         block=(1,1,1)
#     )
# print("func deffed")

# cutlass.cuda.initialize_cuda_context()
# print("context")

# print("running jit compiled")
# kernel_launcher()

# print("precompiled and objdump")
# kernel_compiled = cute.compile(kernel_launcher, options="") # options: --keep-ptx --keep-cubin 
# kernel_compiled()