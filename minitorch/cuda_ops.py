# # type: ignore
# # Currently pyright doesn't support numba.cuda

# from typing import Callable, Optional, TypeVar, Any

# import numba
# from numba import cuda
# from numba.cuda import jit as _jit
# from .tensor import Tensor
# from .tensor_data import (
#     MAX_DIMS,
#     Shape,
#     Storage,
#     Strides,
#     TensorData,
#     broadcast_index,
#     index_to_position,
#     shape_broadcast,
#     to_index,
# )
# from .tensor_ops import MapProto, TensorOps

# FakeCUDAKernel = Any

# # This code will CUDA compile fast versions your tensor_data functions.
# # If you get an error, read the docs for NUMBA as to what is allowed
# # in these functions.

# Fn = TypeVar("Fn")


# def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
#     """device_jit function"""
#     return _jit(device=True, **kwargs)(fn)  # type: ignore


# def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
#     """Jit function"""
#     return _jit(**kwargs)(fn)  # type: ignore


# to_index = device_jit(to_index)
# index_to_position = device_jit(index_to_position)
# broadcast_index = device_jit(broadcast_index)

# THREADS_PER_BLOCK = 32


# class CudaOps(TensorOps):
#     cuda = True

#     @staticmethod
#     def map(fn: Callable[[float], float]) -> MapProto:
#         """See `tensor_ops.py`"""
#         cufn: Callable[[float], float] = device_jit(fn)
#         f = tensor_map(cufn)

#         def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
#             if out is None:
#                 out = a.zeros(a.shape)

#             # Instantiate and run the cuda kernel.
#             threadsperblock = THREADS_PER_BLOCK
#             blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
#             f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
#             return out

#         return ret

#     @staticmethod
#     def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
#         """Zip function"""
#         cufn: Callable[[float, float], float] = device_jit(fn)
#         f = tensor_zip(cufn)

#         def ret(a: Tensor, b: Tensor) -> Tensor:
#             c_shape = shape_broadcast(a.shape, b.shape)
#             out = a.zeros(c_shape)
#             threadsperblock = THREADS_PER_BLOCK
#             blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
#             f[blockspergrid, threadsperblock](  # type: ignore
#                 *out.tuple(), out.size, *a.tuple(), *b.tuple()
#             )
#             return out

#         return ret

#     @staticmethod
#     def reduce(
#         fn: Callable[[float, float], float], start: float = 0.0
#     ) -> Callable[[Tensor, int], Tensor]:
#         """Reduce function"""
#         cufn: Callable[[float, float], float] = device_jit(fn)
#         f = tensor_reduce(cufn)

#         def ret(a: Tensor, dim: int) -> Tensor:
#             out_shape = list(a.shape)
#             out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
#             out_a = a.zeros(tuple(out_shape))

#             threadsperblock = 1024
#             blockspergrid = out_a.size
#             f[blockspergrid, threadsperblock](  # type: ignore
#                 *out_a.tuple(), out_a.size, *a.tuple(), dim, start
#             )

#             return out_a

#         return ret

#     @staticmethod
#     def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
#         """Matrix multiplication"""
#         # Make these always be a 3 dimensional multiply
#         both_2d = 0
#         if len(a.shape) == 2:
#             a = a.contiguous().view(1, a.shape[0], a.shape[1])
#             both_2d += 1
#         if len(b.shape) == 2:
#             b = b.contiguous().view(1, b.shape[0], b.shape[1])
#             both_2d += 1
#         both_2d = both_2d == 2

#         ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
#         ls.append(a.shape[-2])
#         ls.append(b.shape[-1])
#         assert a.shape[-1] == b.shape[-2]
#         out = a.zeros(tuple(ls))

#         # One block per batch, extra rows, extra col
#         blockspergrid = (
#             (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
#             out.shape[0],
#         )
#         threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

#         tensor_matrix_multiply[blockspergrid, threadsperblock](
#             *out.tuple(), out.size, *a.tuple(), *b.tuple()
#         )

#         # Undo 3d if we added it.
#         if both_2d:
#             out = out.view(out.shape[1], out.shape[2])
#         return out


# # Implement


# def tensor_map(
#     fn: Callable[[float], float],
# ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
#     """CUDA higher-order tensor map function. ::

#       fn_map = tensor_map(fn)
#       fn_map(out, ... )

#     Args:
#     ----
#         fn: function mappings floats-to-floats to apply.

#     Returns:
#     -------
#         Tensor map function.

#     """

#     def _map(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         out_size: int,
#         in_storage: Storage,
#         in_shape: Shape,
#         in_strides: Strides,
#     ) -> None:
#         out_index = cuda.local.array(MAX_DIMS, numba.int32)
#         in_index = cuda.local.array(MAX_DIMS, numba.int32)
#         i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#         # TODO: Implement for Task 3.3.
#         # raise NotImplementedError("Need to implement for Task 3.3")

#         if i < out_size:
#             to_index(i, out_shape, out_index)

#             broadcast_index(out_index, out_shape, in_shape, in_index)

#             out_pos = index_to_position(out_index, out_strides)
#             in_pos = index_to_position(in_index, in_strides)

#             out[out_pos] = fn(in_storage[in_pos])

#     return cuda.jit()(_map)  # type: ignore


# def tensor_zip(
#     fn: Callable[[float, float], float],
# ) -> Callable[
#     [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
# ]:
#     """CUDA higher-order tensor zipWith (or map2) function ::

#       fn_zip = tensor_zip(fn)
#       fn_zip(out, ...)

#     Args:
#     ----
#         fn: function mappings two floats to float to apply.

#     Returns:
#     -------
#         Tensor zip function.

#     """

#     def _zip(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         out_size: int,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         b_storage: Storage,
#         b_shape: Shape,
#         b_strides: Strides,
#     ) -> None:
#         out_index = cuda.local.array(MAX_DIMS, numba.int32)
#         a_index = cuda.local.array(MAX_DIMS, numba.int32)
#         b_index = cuda.local.array(MAX_DIMS, numba.int32)
#         i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

#         # TODO: Implement for Task 3.3.
#         # raise NotImplementedError("Need to implement for Task 3.3")

#         # Ensure index is within bounds
#         if i < out_size:
#             to_index(i, out_shape, out_index)

#             broadcast_index(out_index, out_shape, a_shape, a_index)
#             broadcast_index(out_index, out_shape, b_shape, b_index)

#             out_pos = index_to_position(out_index, out_strides)
#             a_pos = index_to_position(a_index, a_strides)
#             b_pos = index_to_position(b_index, b_strides)

#             out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

#     return cuda.jit()(_zip)  # type: ignore


# def _sum_practice(out: Storage, a: Storage, size: int) -> None:
#     r"""This is a practice sum kernel to prepare for reduce.

#     Given an array of length $n$ and out of size $n // \text{blockDIM}$
#     it should sum up each blockDim values into an out cell.

#     $[a_1, a_2, ..., a_{100}]$

#     |

#     $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

#     Note: Each block must do the sum using shared memory!

#     Args:
#     ----
#         out (Storage): storage for `out` tensor.
#         a (Storage): storage for `a` tensor.
#         size (int):  length of a.

#     """  # noqa: D404
#     BLOCK_DIM = 32

#     cache = cuda.shared.array(BLOCK_DIM, numba.float64)
#     i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     pos = cuda.threadIdx.x

#     # TODO: Implement for Task 3.3.
#     # raise NotImplementedError("Need to implement for Task 3.3")
#     if i < size:
#         cache[pos] = a[i]
#     else:
#         cache[pos] = 0.0
#     cuda.syncthreads()

#     stride = 1
#     while stride < BLOCK_DIM:
#         index = 2 * stride * pos
#         if index + stride < BLOCK_DIM:
#             cache[index] += cache[index + stride]
#         stride *= 2
#         cuda.syncthreads()

#     if pos == 0:
#         out[cuda.blockIdx.x] = cache[0]


# jit_sum_practice = cuda.jit()(_sum_practice)


# def sum_practice(a: Tensor) -> TensorData:
#     """sum_practice method"""
#     (size,) = a.shape
#     threadsperblock = THREADS_PER_BLOCK
#     blockspergrid = (size // THREADS_PER_BLOCK) + 1
#     out = TensorData([0.0 for i in range(2)], (2,))
#     out.to_cuda_()
#     jit_sum_practice[blockspergrid, threadsperblock](
#         out.tuple()[0], a._tensor._storage, size
#     )
#     return out


# def tensor_reduce(
#     fn: Callable[[float, float], float],
# ) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
#     """CUDA higher-order tensor reduce function.

#     Args:
#     ----
#         fn: reduction function maps two floats to float.

#     Returns:
#     -------
#         Tensor reduce function.

#     """

#     def _reduce(
#         out: Storage,
#         out_shape: Shape,
#         out_strides: Strides,
#         out_size: int,
#         a_storage: Storage,
#         a_shape: Shape,
#         a_strides: Strides,
#         reduce_dim: int,
#         reduce_value: float,
#     ) -> None:
#         BLOCK_DIM = 1024
#         cache = cuda.shared.array(BLOCK_DIM, numba.float64)
#         out_index = cuda.local.array(MAX_DIMS, numba.int32)
#         out_pos = cuda.blockIdx.x
#         pos = cuda.threadIdx.x

#         # TODO: Implement for Task 3.3.
#         # raise NotImplementedError("Need to implement for Task 3.3")
#         cache[pos] = reduce_value

#         for j in range(a_shape[reduce_dim]):
#             if pos < out_size:
#                 for k in range(len(out_index)):
#                     out_index[k] = 0
#                 out_index[reduce_dim] = j

#                 in_pos = index_to_position(out_index, a_strides)

#                 cache[pos] = fn(cache[pos], a_storage[in_pos])

#         cuda.syncthreads()

#         stride = 1
#         while stride < BLOCK_DIM:
#             index = 2 * stride * pos
#             if index + stride < BLOCK_DIM:
#                 cache[index] = fn(cache[index], cache[index + stride])
#             stride *= 2
#             cuda.syncthreads()

#         if pos == 0:
#             out[out_pos] = cache[0]

#     return jit(_reduce)  # type: ignore


# def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
#     """This is a practice square MM kernel to prepare for matmul.

#     Given a storage `out` and two storage `a` and `b`. Where we know
#     both are shape [size, size] with strides [size, 1].

#     Size is always < 32.

#     Requirements:

#     * All data must be first moved to shared memory.
#     * Only read each cell in `a` and `b` once.
#     * Only write to global memory once per kernel.

#     Compute

#     ```
#      for i:
#          for j:
#               for k:
#                   out[i, j] += a[i, k] * b[k, j]
#     ```

#     Args:
#     ----
#         out (Storage): storage for `out` tensor.
#         a (Storage): storage for `a` tensor.
#         b (Storage): storage for `b` tensor.
#         size (int): size of the square

#     """  # noqa: D404
#     BLOCK_DIM = 32
#     # TODO: Implement for Task 3.3.
#     # raise NotImplementedError("Need to implement for Task 3.3")
#     shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
#     shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)

#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     row = cuda.blockIdx.y * BLOCK_DIM + ty
#     col = cuda.blockIdx.x * BLOCK_DIM + tx

#     result = 0.0

#     for k in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
#         if row < size and k * BLOCK_DIM + tx < size:
#             shared_a[ty, tx] = a[row * size + (k * BLOCK_DIM + tx)]
#         else:
#             shared_a[ty, tx] = 0.0

#         if col < size and k * BLOCK_DIM + ty < size:
#             shared_b[ty, tx] = b[(k * BLOCK_DIM + ty) * size + col]
#         else:
#             shared_b[ty, tx] = 0.0

#         cuda.syncthreads()

#         for n in range(BLOCK_DIM):
#             result += shared_a[ty, n] * shared_b[n, tx]

#         cuda.syncthreads()

#     if row < size and col < size:
#         out[row * size + col] = result


# jit_mm_practice = jit(_mm_practice)


# def mm_practice(a: Tensor, b: Tensor) -> TensorData:
#     """mn_practice method"""
#     (size, _) = a.shape
#     threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
#     blockspergrid = 1
#     out = TensorData([0.0 for i in range(size * size)], (size, size))
#     out.to_cuda_()
#     jit_mm_practice[blockspergrid, threadsperblock](
#         out.tuple()[0], a._tensor._storage, b._tensor._storage, size
#     )
#     return out


# def _tensor_matrix_multiply(
#     out: Storage,
#     out_shape: Shape,
#     out_strides: Strides,
#     out_size: int,
#     a_storage: Storage,
#     a_shape: Shape,
#     a_strides: Strides,
#     b_storage: Storage,
#     b_shape: Shape,
#     b_strides: Strides,
# ) -> None:
#     """
#     Optimized CUDA tensor matrix multiply function using shared memory.

#     Parameters:
#         out, a_storage, b_storage: Global memory arrays for output, input A, and input B.
#         out_shape, a_shape, b_shape: Shapes of the output and input tensors.
#         out_strides, a_strides, b_strides: Strides for indexing the tensors.
#         out_size: Total size of the output tensor.

#     Functionality:
#         - Implements matrix multiplication with broadcasting and shared memory optimization.
#         - Uses CUDA threads and blocks to divide the computation and process blocks in parallel.
#     """
#     # Calculate batch stride (0 if broadcasting a single batch dimension).
#     a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
#     b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
#     # Identify which batch is being processed by this block.
#     batch = cuda.blockIdx.z

#     # Define block size for CUDA shared memory usage.
#     BLOCK_DIM = 32
#     # Allocate shared memory for sub-blocks of A and B matrices.
#     a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
#     b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

#     # Calculate global positions (i, j) in the output matrix.
#     i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#     j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

#     # Calculate local positions (pi, pj) within the thread block.
#     pi = cuda.threadIdx.x
#     pj = cuda.threadIdx.y

#     # Initialize the result for the current thread to 0.
#     result = 0.0

#     # Loop through the shared dimension in chunks of BLOCK_DIM.
#     for k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
#         # Load a block of matrix A into shared memory.
#         if i < a_shape[-2] and k * BLOCK_DIM + pj < a_shape[-1]:
#             a_offset = (
#                 batch * a_batch_stride
#                 + i * a_strides[-2]
#                 + (k * BLOCK_DIM + pj) * a_strides[-1]
#             )
#             a_shared[pi, pj] = a_storage[a_offset]
#         else:
#             a_shared[pi, pj] = 0.0  # Padding for out-of-bound areas.

#         # Load a block of matrix B into shared memory.
#         if j < b_shape[-1] and k * BLOCK_DIM + pi < b_shape[-2]:
#             b_offset = (
#                 batch * b_batch_stride
#                 + (k * BLOCK_DIM + pi) * b_strides[-2]
#                 + j * b_strides[-1]
#             )
#             b_shared[pi, pj] = b_storage[b_offset]
#         else:
#             b_shared[pi, pj] = 0.0  # Padding for out-of-bound areas.

#         # Ensure all threads finish loading shared memory before computation.
#         cuda.syncthreads()

#         # Perform partial dot product for the current block.
#         for n in range(BLOCK_DIM):
#             result += a_shared[pi, n] * b_shared[n, pj]

#         # Synchronize threads again before the next iteration.
#         cuda.syncthreads()

#     # Write the computed result back to global memory, ensuring bounds are respected.
#     if i < out_shape[-2] and j < out_shape[-1]:
#         out_offset = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
#         out[out_offset] = result


# tensor_matrix_multiply = jit(_tensor_matrix_multiply)



##################################################################################################################################

# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs) -> Fn:  # noqa: ANN003
    """device_jit function"""
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn, **kwargs) -> FakeCUDAKernel:  # noqa: ANN001, ANN003
    """Jit function"""
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip function"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce function"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiplication"""
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # if i < out_size:
        #     to_index(i, out_shape, out_index)

        #     broadcast_index(out_index, out_shape, in_shape, in_index)

        #     out_pos = index_to_position(out_index, out_strides)
        #     in_pos = index_to_position(in_index, in_strides)

        #     out[out_pos] = fn(in_storage[in_pos])

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")

        # Ensure index is within bounds
        # if i < out_size:
        #     to_index(i, out_shape, out_index)

        #     broadcast_index(out_index, out_shape, a_shape, a_index)
        #     broadcast_index(out_index, out_shape, b_shape, b_index)

        #     out_pos = index_to_position(out_index, out_strides)
        #     a_pos = index_to_position(a_index, a_strides)
        #     b_pos = index_to_position(b_index, b_strides)

        #     out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """  # noqa: D404
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # if i < size:
    #     cache[pos] = a[i]
    # else:
    #     cache[pos] = 0.0
    # cuda.syncthreads()

    # stride = 1
    # while stride < BLOCK_DIM:
    #     index = 2 * stride * pos
    #     if index + stride < BLOCK_DIM:
    #         cache[index] += cache[index + stride]
    #     stride *= 2
    #     cuda.syncthreads()

    # if pos == 0:
    #     out[cuda.blockIdx.x] = cache[0]

    if i < size:
        val = float(a[i])
        cache[pos] = val
        cuda.syncthreads()
    else:
        cache[pos] = 0.0

    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (j * 2) == 0:
                cache[pos] += cache[pos + j]
            cuda.syncthreads()

        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """sum_practice method"""
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        # raise NotImplementedError("Need to implement for Task 3.3")
        # cache[pos] = reduce_value

        # for j in range(a_shape[reduce_dim]):
        #     if pos < out_size:
        #         for k in range(len(out_index)):
        #             out_index[k] = 0
        #         out_index[reduce_dim] = j

        #         in_pos = index_to_position(out_index, a_strides)

        #         cache[pos] = fn(cache[pos], a_storage[in_pos])

        # cuda.syncthreads()

        # stride = 1
        # while stride < BLOCK_DIM:
        #     index = 2 * stride * pos
        #     if index + stride < BLOCK_DIM:
        #         cache[index] = fn(cache[index], cache[index + stride])
        #     stride *= 2
        #     cuda.syncthreads()

        # if pos == 0:
        #     out[out_pos] = cache[0]


        cache[pos] = reduce_value

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < a_shape[reduce_dim]:
                a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[a]
                cuda.syncthreads()

                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (j * 2) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                    cuda.syncthreads()
                    x += 1

                if pos == 0:
                    out[o] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """  # noqa: D404
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # raise NotImplementedError("Need to implement for Task 3.3")
    # shared_a = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)
    # shared_b = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), dtype=numba.float32)

    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y
    # row = cuda.blockIdx.y * BLOCK_DIM + ty
    # col = cuda.blockIdx.x * BLOCK_DIM + tx

    # result = 0.0

    # for k in range((size + BLOCK_DIM - 1) // BLOCK_DIM):
    #     if row < size and k * BLOCK_DIM + tx < size:
    #         shared_a[ty, tx] = a[row * size + (k * BLOCK_DIM + tx)]
    #     else:
    #         shared_a[ty, tx] = 0.0

    #     if col < size and k * BLOCK_DIM + ty < size:
    #         shared_b[ty, tx] = b[(k * BLOCK_DIM + ty) * size + col]
    #     else:
    #         shared_b[ty, tx] = 0.0

    #     cuda.syncthreads()

    #     for n in range(BLOCK_DIM):
    #         result += shared_a[ty, n] * shared_b[n, tx]

    #     cuda.syncthreads()

    # if row < size and col < size:
    #     out[row * size + col] = result

    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i >= size or j >= size:
        return

    a_shared[i, j] = a[size * i + j]
    b_shared[i, j] = b[size * i + j]
    cuda.syncthreads()

    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]

    out[size * i + j] = accum


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """mn_practice method"""
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """Optimized CUDA tensor matrix multiply function using shared memory.

    Parameters:
        out, a_storage, b_storage: Global memory arrays for output, input A, and input B.
        out_shape, a_shape, b_shape: Shapes of the output and input tensors.
        out_strides, a_strides, b_strides: Strides for indexing the tensors.
        out_size: Total size of the output tensor.

    Functionality:
        - Implements matrix multiplication with broadcasting and shared memory optimization.
        - Uses CUDA threads and blocks to divide the computation and process blocks in parallel.
    """
    # Calculate batch stride (0 if broadcasting a single batch dimension).
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Identify which batch is being processed by this block.
    batch = cuda.blockIdx.z

    # Define block size for CUDA shared memory usage.
    BLOCK_DIM = 32
    # # Allocate shared memory for sub-blocks of A and B matrices.
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # Calculate global positions (i, j) in the output matrix.
    # i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # # Calculate local positions (pi, pj) within the thread block.
    # pi = cuda.threadIdx.x
    # pj = cuda.threadIdx.y

    # # Initialize the result for the current thread to 0.
    # result = 0.0

    # # Loop through the shared dimension in chunks of BLOCK_DIM.
    # for k in range((a_shape[-1] + BLOCK_DIM - 1) // BLOCK_DIM):
    #     # Load a block of matrix A into shared memory.
    #     if i < a_shape[-2] and k * BLOCK_DIM + pj < a_shape[-1]:
    #         a_offset = (
    #             batch * a_batch_stride
    #             + i * a_strides[-2]
    #             + (k * BLOCK_DIM + pj) * a_strides[-1]
    #         )
    #         a_shared[pi, pj] = a_storage[a_offset]
    #     else:
    #         a_shared[pi, pj] = 0.0  # Padding for out-of-bound areas.

    #     # Load a block of matrix B into shared memory.
    #     if j < b_shape[-1] and k * BLOCK_DIM + pi < b_shape[-2]:
    #         b_offset = (
    #             batch * b_batch_stride
    #             + (k * BLOCK_DIM + pi) * b_strides[-2]
    #             + j * b_strides[-1]
    #         )
    #         b_shared[pi, pj] = b_storage[b_offset]
    #     else:
    #         b_shared[pi, pj] = 0.0  # Padding for out-of-bound areas.

    #     # Ensure all threads finish loading shared memory before computation.
    #     cuda.syncthreads()

    #     # Perform partial dot product for the current block.
    #     for n in range(BLOCK_DIM):
    #         result += a_shared[pi, n] * b_shared[n, pj]

    #     # Synchronize threads again before the next iteration.
    #     cuda.syncthreads()

    # # Write the computed result back to global memory, ensuring bounds are respected.
    # if i < out_shape[-2] and j < out_shape[-1]:
    #     out_offset = batch * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]
    #     out[out_offset] = result

    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    batch = cuda.blockIdx.z
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    accum = 0.0

    for k_start in range(0, a_shape[2], BLOCK_DIM):
        k = k_start + pj
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[
                a_batch_stride * batch + a_strides[1] * i + a_strides[2] * k
            ]
        k = k_start + pi
        if j < b_shape[2] and k < b_shape[1]:
            b_shared[pi, pj] = b_storage[
                b_batch_stride * batch + b_strides[1] * k + b_strides[2] * j
            ]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            if (k_start + k) < a_shape[2]:
                accum += a_shared[pi, k] * b_shared[k, pj]

        cuda.syncthreads()

    if i < out_shape[1] and j < out_shape[2]:
        out[out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j] = accum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
