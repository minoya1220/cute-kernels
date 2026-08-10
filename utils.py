import cutlass.cute as cute
from collections.abc import Callable

def warp_reduction_partial(value: cute.typing.Numeric, op: Callable, width: int) -> cute.typing.Numeric:
    assert 1 <= width and width <= 32, f"width must be in [1,32], got {width}"
    assert (width & (width - 1)) == 0, f"width must be a power of 2, got {width}"
    result = value
    for i in range(width.bit_length() - 1):
        result = op(result, cute.arch.shuffle_sync_bfly(result,1 << i))
    return result