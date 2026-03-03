import cutlass
import cutlass.cute as cute

@cute.kernel
def softmax_fwd_kernel(gX: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bidx * bdim + tidx # global tid

    pass

@cute.kernel
def softmax_bwd_kernel(gY: cute.Tensor, gdY: cute.Tensor, gdX: cute.Tensor):
    
    pass

@cute.jit
def softmax_fwd_kernel(T):
    block_size = 256

    pass
