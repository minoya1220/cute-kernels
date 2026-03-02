import cutlass
import cutlass.cute as cute

@cute.kernel
def softmax_kernel(data, d_H, d_B):
    pass

@cute.jit
def softmax(T):
    pass
