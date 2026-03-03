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

@cute.jit
def vector_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # not sure what the m in mA and mB refers to, could be for "matrix" essentially represents raw input tensor
    
    block_size = 256

    m, n = mA.shape

    noob_kernel = vector_add_kernel_noob(mA, mB, mC)

    noob_kernel.launch(
        grid=((m*n + block_size - 1) // block_size, 1, 1),
        block=(block_size, 1, 1)
    )

M, N = 16384, 8192

A = torch.zeros(M,N, device='cuda', dtype=toch.bfloat16)
B = torch.zeros(M,N, device='cuda', dtype=toch.bfloat16)
C = torch.zeros(M,N, device='cuda', dtype=toch.bfloat16)

total_elem = M * N * 3

# converts torch tensors to CuTe tensors
a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

vector_add_ = cute.compile(vector_add, a_, b_, c_)

vector_add_(a_, b_, c_)
print("test run successful")




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