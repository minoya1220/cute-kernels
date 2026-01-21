print("file launched")
import cutlass
import cutlass.cute as cute
print("imported")

@cute.kernel
def kernel():
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        cute.printf("dwadowjad")

@cute.jit
def kernel_launcher():
    cute.printf("launching kernel...")

    kernel().launch(
        grid=(1,1,1),
        block=(1,1,1)
    )
print("func deffed")

cutlass.cuda.initialize_cuda_context()
print("context")

print("running jit compiled")
kernel_launcher()

print("precompiled and objdump")
kernel_compiled = cute.compile(kernel_launcher, options="") # options: --keep-ptx --keep-cubin 
kernel_compiled()