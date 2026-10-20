from __future__ import absolute_import
import numpy as np
import numpy.typing as npt
import itertools
import math
from functools import lru_cache
from typing import Any, Iterable, List, Tuple, Union

IntArray = npt.NDArray[np.int_]
UIntArray = npt.NDArray[np.uint64]
BoolArray = npt.NDArray[np.bool_]


@lru_cache(maxsize=16)
def _reversed_base(base: Tuple[Any, ...]) -> IntArray:
    if len(base)>1:
        return np.hstack((1, np.flipud(np.asarray(base)[1:])))
    else:
        return np.ones(len(base), dtype=int)
def reversed_base(base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    return _reversed_base(tuple(base))

@lru_cache(maxsize=16)
def _radix_converter(base: Tuple[Any, ...]) -> UIntArray:
    as_object_array = np.flipud(np.multiply.accumulate(
        reversed_base(base).astype(dtype=object)
    ))
    #We are concerned about integer overflow. To avoid this we perform the accumulation in Python, outside of Numpy.
    as_object_list = tuple(map(int, as_object_array.tolist()))
    if np.can_cast(np.min_scalar_type(as_object_list), np.uintp):
        return np.array(as_object_list, dtype=np.uintp)
    else:
        return as_object_array

def radix_converter(base: Union[Tuple[Any, ...], IntArray]) -> UIntArray:
    return _radix_converter(tuple(base))

@lru_cache(maxsize=16)
def _uniform_base_test(base: Tuple[Any, ...]) -> bool:
    return np.array_equiv(base[0], base)
def uniform_base_test(base: Union[Tuple[Any, ...], IntArray]) -> bool:
    return _uniform_base_test(tuple(base))

@lru_cache(maxsize=16)
def _binary_base_test(base: Tuple[Any, ...]) -> bool:
    return np.array_equiv(2, base)
def binary_base_test(base: Union[Tuple[Any, ...], IntArray]) -> bool:
    return _binary_base_test(tuple(base))


def flip_array_last_axis(m: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return m[..., ::-1]

def to_bits(integers: Union[int, IntArray], mantissa: int, sanity_check: bool = False) -> BoolArray:
    """
    Shape: (...,) -> (..., mantissa).
    Threads over leading axes and expands each scalar into a trailing axis of bits of length `mantissa`.
    """
    if sanity_check:
        min_mantissa = int(np.amax(integers)).bit_length()
        if min_mantissa>mantissa:
            print("ERROR: Mantissa is too small to accommodate an integer, overriding.")
            mantissa = min_mantissa
    if mantissa<=8:
        bytes_array = np.expand_dims(np.asarray(integers, np.uint8), axis=-1)
    elif mantissa<=16:
        bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers,np.uint16), axis=-1).view(np.uint8))
    elif mantissa <= 32:
        bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers, np.uint32), axis=-1).view(np.uint8))
    elif mantissa <= 64:
        bytes_array = flip_array_last_axis(np.expand_dims(np.asarray(integers, np.uint64), axis=-1).view(np.uint8))
    else:
        # arbitrary precision: Python integers are the reference (a numpy scalar must be converted first)
        mantissa_for_bytes = -(-mantissa // 8)
        to_bytes_ufunc = np.vectorize(lambda x: np.frombuffer(int(x).to_bytes(length=mantissa_for_bytes, byteorder='big'), dtype=np.uint8),
                                      otypes=[np.uint8], signature='()->(n)')
        bytes_array = to_bytes_ufunc(np.asarray(integers, dtype=object))
    return np.unpackbits(bytes_array, axis=-1, bitorder='big')[..., -mantissa:]

# import numba
# @numba.vectorize([
#     numba.int64(numba.uint8, numba.uint8),
#     numba.int64(numba.uint8, numba.int64),
#     numba.int64(numba.uint, numba.uint),
#     numba.int64(numba.uint, numba.int64),
#     numba.int64(numba.int64, numba.int64)
# ], nopython = True)
# def pack_a_bit(byte, bit):
#     #return byte << 1 | bit
#     #one = 1
#     return np.bitwise_or(np.left_shift(byte, 1), bit)
#
# def from_bits(smooshed_bit_array):
#     return pack_a_bit.reduce(smooshed_bit_array, axis=-1)

def from_littleordered_bytes(byte_array: npt.NDArray[np.uint8]) -> npt.NDArray[Any]:
    """Shape: (..., nbytes) -> (...,) of Python integers (object dtype); every leading axis is preserved."""
    as_array = np.asarray(byte_array, dtype=np.uint8)
    rows = as_array.reshape((-1, as_array.shape[-1]))
    values = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows.tolist()):
        values[i] = int.from_bytes(bytes(row), byteorder='little', signed=False)
    return values.reshape(as_array.shape[:-1])

def from_bits(smooshed_bit_array: BoolArray) -> IntArray:
    """
    Shape: (..., mantissa) -> (...,).
    Threads over leading axes and collapses the trailing bit axis back into integers.
    """
    mantissa = smooshed_bit_array.shape[-1]
    if mantissa > 0:
        ready_for_viewing = np.packbits(flip_array_last_axis(smooshed_bit_array), axis=-1, bitorder='little')
        final_dimension = ready_for_viewing.shape[-1]
        if mantissa<=8:
            return np.squeeze(ready_for_viewing, axis=-1)
        elif mantissa<=16:
            return np.squeeze(ready_for_viewing.view(np.uint16), axis=-1)
        elif mantissa <= 32:
            pad_size = 4-final_dimension
            if pad_size == 0:
                return np.squeeze(ready_for_viewing.view(np.uint32), axis=-1)
            else:
                npad = [(0, 0)] * ready_for_viewing.ndim
                npad[-1] = (0, pad_size)
                return np.squeeze(np.ascontiguousarray(
                    np.pad(ready_for_viewing, pad_width=npad, mode='constant', constant_values=0)
                ).view(np.uint32), axis=-1)
        elif mantissa <= 64:
            pad_size = 8-final_dimension
            if pad_size == 0:
                return np.squeeze(ready_for_viewing.view(np.uint64), axis=-1)
            else:
                npad = [(0, 0)] * ready_for_viewing.ndim
                npad[-1] = (0, pad_size)
                return np.squeeze(np.ascontiguousarray(
                    np.pad(ready_for_viewing, pad_width=npad, mode='constant', constant_values=0)
                ).view(np.uint64), axis=-1)
        else:   # beyond 64 bits: Python integers, exact
            return from_littleordered_bytes(ready_for_viewing)
    else:
        return np.zeros(smooshed_bit_array.shape[:-2], dtype=int)

def _from_digits(digits_array: IntArray, base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    return np.matmul(digits_array, radix_converter(base))

def from_digits(digits_array: Union[List[Any], IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    """
    Shape: (..., k) -> (...,).
    Threads over leading axes and collapses the trailing digit axis using the provided base.
    """
    digits_array_as_array = np.asarray(digits_array)
    if min(digits_array_as_array.shape) > 0:
        if binary_base_test(base):
            return from_bits(digits_array_as_array)
        else:
            return _from_digits(digits_array_as_array, base)
    else:
        return digits_array_as_array.reshape(digits_array_as_array.shape[:-1])




def array_from_string(string_array: Union[str, List[str], IntArray]) -> IntArray:
    as_string_array = np.asarray(string_array, dtype=str)
    return np.fromiter(itertools.chain.from_iterable(as_string_array.ravel()), np.uint).reshape(
        as_string_array.shape + (-1,))


def from_string_digits(string_digits_array: Union[str, List[str], IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    return from_digits(array_from_string(string_digits_array), base)


@lru_cache(maxsize=16)
def _unravel_index_applies(base: Tuple[Any, ...]) -> bool:
    """np.unravel_index needs at most 32 dimensions and a total size that fits in intp."""
    return len(base) <= 32 and math.prod(int(b) for b in base) <= np.iinfo(np.intp).max


def _to_digits(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    if _unravel_index_applies(tuple(int(b) for b in base)):
        return np.stack(np.unravel_index(np.asarray(integer, dtype=np.intp), base), axis=-1)
    else:
        return _to_digits_divmod(integer, base)

def _to_digits_divmod(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    """Digit extraction by repeated divmod, least significant digit first; works on object arrays of Python
    integers, so there is no width limit."""
    arrays = []
    x = np.array(integer, copy=True)
    if x.dtype.kind in 'ui' and not _unravel_index_applies(tuple(int(b) for b in base)):
        x = x.astype(object)
    for b in np.flipud(base).flat:
        b = int(b)
        # np.divmod has no object loop; floor_divide and remainder do
        x, remainder = np.floor_divide(x, b), np.remainder(x, b)
        arrays.append(remainder)
    return flip_array_last_axis(np.stack(arrays, axis=-1))

def to_digits(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray], sanity_check: bool = False) -> IntArray:
    """
    Shape: (...,) -> (..., k).
    Threads over leading axes and expands scalars into digit vectors of length len(base).
    """
    if sanity_check:
        does_it_fit = int(np.amax(integer)) < math.prod(int(b) for b in base)
        assert does_it_fit, "Base is too small to accommodate such large integers."
    if binary_base_test(base):
        return to_bits(integer, len(base))
    else:
        return _to_digits(integer, base)


def array_to_string(digits_array: IntArray) -> Union[str, List[str]]:
    before_string_joining = np.asarray(digits_array, dtype=str)
    raw_shape = before_string_joining.shape
    return np.array(list(
        map(''.join, before_string_joining.reshape((-1, raw_shape[-1])).astype(str))
    )).reshape(raw_shape[:-1]).tolist()


def to_string_digits(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray]) -> Union[str, List[str]]:
    return array_to_string(to_digits(integer, base))

# def bitarrays_to_ints(bit_array):
#     bit_array_as_array = np.asarray(bit_array)
#     shape = bit_array_as_array.shape
#     (numrows, numcolumns) = shape[-2:]
#     # return from_digits(
#     #     from_bits(bit_array_as_array),
#     #     np.broadcast_to(2**numcolumns, numrows))
#     return from_bits(bit_array_as_array.reshape(shape[:-2]+(numrows * numcolumns,)))

@lru_cache(maxsize=None)
def _bitarray_to_int(bit_array: Tuple[Tuple[bool, ...], ...]) -> int:
    if len(bit_array):
        bit_array_as_array = np.asarray(bit_array, dtype=bool)
        (numrows, numcolumns) = bit_array_as_array.shape
        rows_basis = np.repeat(2**numcolumns, numrows)
        first_conversion = from_bits(bit_array_as_array)
        return _from_digits(first_conversion, rows_basis)
    else:
        return 0
    # return from_bits(bit_array.ravel()).tolist()

def bitarray_to_int(bit_array: BoolArray) -> int:
    return _bitarray_to_int(tuple(map(tuple, bit_array)))

# def ints_to_bitarrays(integer, numcolumns):
#     # numrows = -np.floor_divide(-np.log1p(integer), np.log(2**numcolumns)).astype(int).max() #Danger border case
#     # return np.reshape(to_bits(integer, numrows * numcolumns), np.asarray(integer).shape + (numrows, numcolumns))
#     return to_bits(integer, numcolumns)











