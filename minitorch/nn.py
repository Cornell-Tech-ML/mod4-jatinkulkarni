from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

reduce_max = FastOps.reduce(operators.max, float("-inf"))


# List of functions in this file:
# ✅ avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")

    new_height = height // kh
    new_width = width // kw

    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    transposed = reshaped.permute(0, 1, 2, 4, 3, 5)
    tiled = transposed.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width



def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute 2D average pooling.
    
    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    
    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tiled_tensor, new_height, new_width = tile(input, kernel)

    pooled_tensor = tiled_tensor.mean(dim=4)
    reduced_tensor = pooled_tensor.contiguous().view(batch, channel, new_height, new_width)

    return reduced_tensor

# TODO: Implement for Task 4.3.

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor"""
    maximum_values = reduce_max(input, dim)
    return maximum_values == 1

class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward method for Max"""
        dim_int = int(dim._tensor._storage[0]) if isinstance(dim, Tensor) else int(dim)

        ctx.save_for_backward(t1, t1._ensure_tensor(dim_int))
        return reduce_max(t1, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward method for Max"""
        t1, dim = ctx.saved_values
        dim_int = int(dim._tensor._storage[0]) if isinstance(dim, Tensor) else int(dim)
        
        return (argmax(t1, dim_int) * grad_output, t1._ensure_tesnor(dim_int))


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    max_val = reduce_max(input, dim)
    exp_input = (input - max_val).exp()
    sum_exp = exp_input.sum(dim=dim)
    return exp_input / sum_exp

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    max_val = reduce_max(input, dim)
    log_sum_exp = (input - max_val).exp().sum(dim=dim).log()
    return input - max_val - log_sum_exp

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D."""
    tiled_tensor, new_height, new_width = tile(input, kernel)
    pooled_tensor = max(tiled_tensor, dim=-1)
    return pooled_tensor.view(input.shape[0], input.shape[1], new_height, new_width)

def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off."""
    if ignore or rate <= 0:
        return input

    mask = rand(input.shape) > rate
    return input * mask

