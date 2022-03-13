from __future__ import absolute_import
import numpy as np
import itertools
import operator
from functools import lru_cache

@lru_cache(maxsize=16)
def _reversed_base(base):
    if len(base)>1:
        return np.hstack((1, np.flipud(np.asarray(base)[1:])))
    else:
        return np.ones(len(base), dtype=int)
def reversed_base(base):
    return _reversed_base(tuple(base))

@lru_cache(maxsize=16)
def _radix_converter(base):
    as_object_array = np.flipud(np.multiply.accumulate(
        reversed_base(base).astype(dtype=object)
    ))
    #We are concerned about integer overflow. To avoid this we perform the accumulation in Python, outside of Numpy.
    as_object_list = tuple(map(int, as_object_array.tolist()))
    if np.can_cast(np.min_scalar_type(as_object_list), np.uintp):
        return np.array(as_object_list, dtype=np.uintp)
    else:
        return as_object_array

def radix_converter(base):
    return _radix_converter(tuple(base))

@lru_cache(maxsize=16)
def _uniform_base_test(base):
    return np.array_equiv(base[0], base)
def uniform_base_test(base):
    return _uniform_base_test(tuple(base))

@lru_cache(maxsize=16)
def _binary_base_test(base):
    return np.array_equiv(2, base)
def binary_base_test(base):
    return _binary_base_test(tuple(base))


def flip_array_last_axis(m):
    return m[..., ::-1]

def to_bits(integers, mantissa, sanity_check=False):
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

def from_littleordered_bytes(byte_array):
    #Assumes numpy array
    if byte_array.ndim == 1:
        return np.asarray(int.from_bytes(byte_array.tolist(), byteorder='little', signed=False))
    else:
        return np.vstack(list(map(from_littleordered_bytes, byte_array)))

def from_bits(smooshed_bit_array):
    mantissa = smooshed_bit_array.shape[-1]
    if mantissa > 0:
        # possible_mantissas = np.array([8,16,32,64])
        # effective_mantissa = possible_mantissas.compress(np.floor_divide(possible_mantissas, mantissa))[0]
        ready_for_viewing = np.packbits(flip_array_last_axis(smooshed_bit_array), axis=-1, bitorder='little')
        # return from_littleordered_bytes(ready_for_viewing)
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
            # print("Byte array in little endian ordering: ", ready_for_viewing)
            return np.squeeze(from_littleordered_bytes(ready_for_viewing), axis=-1)
    else:
        return np.zeros(smooshed_bit_array.shape[:-2], dtype=int)

def _from_digits(digits_array, base):
    return np.matmul(digits_array, radix_converter(base))

def from_digits(digits_array, base):
    digits_array_as_array = np.asarray(digits_array)
    if min(digits_array_as_array.shape) > 0:
        if binary_base_test(base):
            return from_bits(digits_array_as_array)
        else:
            return _from_digits(digits_array_as_array, base)
    else:
        return digits_array_as_array.reshape(digits_array_as_array.shape[:-1])




def array_from_string(string_array):
    as_string_array = np.asarray(string_array, dtype=str)
    return np.fromiter(itertools.chain.from_iterable(as_string_array.ravel()), np.uint).reshape(
        as_string_array.shape + (-1,))


def from_string_digits(string_digits_array, base):
    return from_digits(array_from_string(string_digits_array), base)


def _to_digits(integer, base):
    if len(base)<=32:
        return np.stack(np.unravel_index(np.asarray(integer, dtype=np.intp), base), axis=-1)
    else:
        arrays = []
        x = np.array(integer, copy=True)
        #print(reversed_base(base))
        for b in reversed(base):
            x, remainder = np.divmod(x, b)
            arrays.append(remainder)
        return flip_array_last_axis(np.stack(arrays, axis=-1))

def to_digits(integer, base, sanity_check=False):
    if sanity_check:
        does_it_fit = np.amax(integer) < np.multiply.reduce(np.flipud(base).astype(dtype=np.ulonglong))
        assert does_it_fit, "Base is too small to accommodate such large integers."
    if binary_base_test(base):
        return to_bits(integer, len(base))
    else:
        return _to_digits(integer, base)


def array_to_string(digits_array):
    before_string_joining = np.asarray(digits_array, dtype=str)
    raw_shape = before_string_joining.shape
    return np.array(list(
        map(''.join, before_string_joining.reshape((-1, raw_shape[-1])).astype(str))
    )).reshape(raw_shape[:-1]).tolist()


def to_string_digits(integer, base):
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
def _bitarray_to_int(bit_array):
    if len(bit_array):
        bit_array_as_array = np.asarray(bit_array, dtype=bool)
        (numrows, numcolumns) = bit_array_as_array.shape
        rows_basis = np.repeat(2**numcolumns, numrows)
        first_conversion = from_bits(bit_array_as_array)
        return _from_digits(first_conversion, rows_basis)
    else:
        return 0
    # return from_bits(bit_array.ravel()).tolist()

def bitarray_to_int(bit_array):
    return _bitarray_to_int(tuple(map(tuple, bit_array)))

# def ints_to_bitarrays(integer, numcolumns):
#     # numrows = -np.floor_divide(-np.log1p(integer), np.log(2**numcolumns)).astype(int).max() #Danger border case
#     # return np.reshape(to_bits(integer, numrows * numcolumns), np.asarray(integer).shape + (numrows, numcolumns))
#     return to_bits(integer, numcolumns)

if __name__ == '__main__':
    print(to_digits([3,5,12,100], base=np.hstack((np.repeat(1,32),(3,4,5)))))
    print(to_digits([3, 5, 12, 100], base=np.hstack((np.repeat(1, 32), (3, 4, 5)))).shape)

    integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    base = (2, 3, 2, 3, 4, 2, 2, 3, 2)
    # digits_array = to_digits(integers, base)
    # print(to_digits(integers, base))
    # print(to_string_digits(integers, base))
    # print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    integers = np.ravel(integers)
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    # integers = 1237
    # print(to_digits(integers,base))
    # print(from_digits(to_digits(integers,base),base))
    #
    # integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    # base =np.broadcast_to(2,11)
    # digits_array = to_digits(integers, base)
    # print(to_digits(integers, base))
    # print(to_string_digits(integers, base))
    # print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    # print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    # integers = 1237
    # print(to_digits(integers,base))
    # print(from_digits(to_digits(integers,base),base))
    #
    # print(from_digits([],base))
    # print(to_digits([], base))
    #print(int_to_bitarray(integers, 11))
    # as_bits = to_bits(integers, 11)
    # print("Effect of encoding to bits: ", as_bits, as_bits.shape)
    # print("Effect of bit encoding and decoding: ", from_bits(as_bits))
    print("Testing bit arithmetic for n-dimensional arrays...")
    print(np.array_equiv(integers, from_bits(to_bits(integers, 11))))

    print("Testing for arithmetic overflow.")
    integers = [5,40,12312312,2**63,2**64,2**65]
    print("Integers to encode: ", integers)
    print("Testing sanity_check option:")
    encoded_integers = to_bits(integers, 65, sanity_check=True)
    encoded_integers = to_bits(integers, 66)
    # print("Encoded integers: ", encoded_integers)
    recoded_integers = from_bits(encoded_integers)
    print("Decoded integers: ", recoded_integers)
    print(np.array_equiv(integers, recoded_integers))





    # def to_bytes_backwards(integers):
    #     before = np.asarray(integers)
    #     after = before.view(np.uint8)
    #     return after.reshape(before.shape+(-1,))
    #
    # def to_bits_backwards(integers):
    #     return np.unpackbits(to_bytes_backwards(integers), axis=-1, bitorder='big')
    #
    # def from_bits_backwards(bit_array, dtype):
    #     # np.packbits(np.flip(bit_array,axis=-1), axis=-1, bitorder='little')
    #     return np.squeeze(np.packbits(bit_array, axis=-1, bitorder='big').view(dtype),axis=-1)











