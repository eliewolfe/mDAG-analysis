from __future__ import absolute_import
import numpy as np
import numpy.typing as npt
import itertools
# import operator
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
        min_mantissa = int.bit_length(np.amax(integers))
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
        mantissa_for_bytes = np.floor_divide(mantissa, 8) + 1
        # to_bytes_ufunc = np.frompyfunc(lambda x: np.frombuffer(int.to_bytes(x, length=mantissa_for_bytes, byteorder='big'), dtype=np.uint8), 1, 1)
        to_bytes_ufunc = np.vectorize(lambda x: np.frombuffer(int.to_bytes(x, length=mantissa_for_bytes, byteorder='big'), dtype=np.uint8), signature='()->(n)')
        bytes_array = to_bytes_ufunc(integers)
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

def from_littleordered_bytes(byte_array: npt.NDArray[np.uint8]) -> UIntArray:
    #Assumes numpy array
    if byte_array.ndim == 1:
        return np.asarray(int.from_bytes(byte_array.tolist(), byteorder='little', signed=False))
    else:
        return np.vstack(list(map(from_littleordered_bytes, byte_array)))

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
        elif mantissa > 64:
            print("Warning: Integers exceeding 2^64, possible overflow errors.")
            return np.squeeze(from_littleordered_bytes(ready_for_viewing), axis=-1)
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


def _to_digits(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    if len(base)<=32:
        return np.stack(np.unravel_index(np.asarray(integer, dtype=np.intp), base), axis=-1)
    else:
        return _to_digits_numba(integer, base)

def _to_digits_numba(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray]) -> IntArray:
    arrays = []
    x = np.array(integer, copy=True)
    #print(reversed_base(base))
    for b in np.flipud(base).flat:
        x, remainder = np.divmod(x, b)
        arrays.append(remainder)
    return flip_array_last_axis(np.stack(arrays, axis=-1))

def to_digits(integer: Union[int, IntArray], base: Union[Tuple[Any, ...], IntArray], sanity_check: bool = False) -> IntArray:
    """
    Shape: (...,) -> (..., k).
    Threads over leading axes and expands scalars into digit vectors of length len(base).
    """
    if sanity_check:
        does_it_fit = np.amax(integer) < np.multiply.reduce(np.flipud(base).astype(dtype=np.ulonglong))
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


try:
    from numba import guvectorize  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    _NUMBA_TO_BITS_CACHE: dict[int, Any] = {}

    def get_numba_to_bits(mantissa: int):
        """Return a guvectorized to_bits for the given mantissa, cached by mantissa length."""
        if mantissa not in _NUMBA_TO_BITS_CACHE:
            @guvectorize(['void(uint64[:], uint64[:], uint64[:])'], '(),(m)->(m)', nopython=True)
            def _numba_to_bits(integer, dummy, out):
                val = integer[0]
                for j in range(out.shape[0] - 1, -1, -1):
                    out[j] = val & 1
                    val >>= 1
            _NUMBA_TO_BITS_CACHE[mantissa] = _numba_to_bits
        return _NUMBA_TO_BITS_CACHE[mantissa]

    @guvectorize(['void(uint64[:], uint64[:])'], '(m)->()', nopython=True)
    def numba_from_bits(bits, out):
        val = 0
        for j in range(bits.shape[0]):
            val = (val << 1) | bits[j]
        out[0] = val

    @guvectorize(['void(int64[:], int64[:], int64[:])'], '(n),(n)->()', nopython=True)
    def numba_from_digits(digits, base, out):
        val = 0
        for j in range(digits.shape[0]):
            val = val * base[j] + digits[j]
        out[0] = val

    @guvectorize(['void(int64[:], int64[:], int64[:])'], '(),(n)->(n)', nopython=True)
    def numba_to_digits(integer, base, out):
        val = integer[0]
        for j in range(base.shape[0] - 1, -1, -1):
            b = base[j]
            out[j] = val % b
            val //= b

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    print("Testing to_digits/from_digits, to_bits/from_bits shapes...")
    base = (2, 3, 2, 3, 4)
    data = rng.integers(0, 4, size=(2, 3), dtype=np.int64)
    digits = to_digits(data, base)
    back = from_digits(digits, base)
    print("Digits shape", digits.shape, "roundtrip ok:", np.array_equal(data, back))

    bits = to_bits(data, 11)
    back_bits = from_bits(bits)
    print("Bits shape", bits.shape, "roundtrip ok:", np.array_equal(data, back_bits))

    if NUMBA_AVAILABLE:
        base_arr = np.asarray(base, dtype=np.int64)
        numba_dig = numba_to_digits(data, base_arr)
        numba_back = numba_from_digits(numba_dig, base_arr)
        print("Numba digits roundtrip ok:", np.array_equal(data, numba_back))

        bits_shape = bits.shape[-1]
        numba_to_bits_fn = get_numba_to_bits(bits_shape)
        numba_bits = numba_to_bits_fn(np.asarray(data, dtype=np.uint64), np.empty(bits_shape, dtype=np.uint64))
        numba_back_bits = numba_from_bits(numba_bits, np.empty(1, dtype=np.uint64))
        print("Numba bits roundtrip ok:", np.array_equal(data, numba_back_bits))

    # Simple timing comparison
    big_data = rng.integers(0, 1 << 16, size=(1000,), dtype=np.int64)
    base_small = (2, 3, 2, 3)
    import time
    t0 = time.time()
    _ = to_digits(big_data, base_small)
    t1 = time.time()
    print("Python to_digits 1k elapsed:", t1 - t0)
    if NUMBA_AVAILABLE:
        _ = numba_to_digits(big_data, np.asarray(base_small, dtype=np.int64))
        t2 = time.time()
        print("Numba to_digits 1k elapsed:", t2 - t1)











