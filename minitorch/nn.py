from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
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
