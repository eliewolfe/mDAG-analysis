from __future__ import absolute_import
import numpy as np
import itertools
from functools import lru_cache

@lru_cache(maxsize=16)
def _radix_converter(base):
    return np.flip(np.multiply.accumulate(
        np.hstack((1, np.flip(base[1:]))).astype(dtype=np.ulonglong)
    ))
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


def to_bits(integers, mantissa):
    if mantissa<=8:
        bytes_array = np.expand_dims(np.asarray(integers,np.uint8), axis=-1)
    elif mantissa<=16:
        bytes_array = np.flip(np.expand_dims(np.asarray(integers,np.uint16), axis=-1).view(np.uint8),axis=-1)
    elif mantissa <= 32:
        bytes_array = np.flip(np.expand_dims(np.asarray(integers, np.uint32), axis=-1).view(np.uint8), axis=-1)
    elif mantissa <= 64:
        bytes_array = np.flip(np.expand_dims(np.asarray(integers, np.uint64), axis=-1).view(np.uint8), axis=-1)
    return np.unpackbits(bytes_array, axis=-1, bitorder='big')[...,-mantissa:]

def from_bits(smooshed_bit_array):
    mantissa = np.asarray(smooshed_bit_array).shape[-1]
    possible_mantissas = np.array([8,16,32,64])
    effective_mantissa = possible_mantissas.compress(np.floor_divide(possible_mantissas, mantissa))[0]
    ready_for_viewing = np.packbits(np.flip(smooshed_bit_array, axis=-1), axis=-1, bitorder='little')
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

def _from_digits(digits_array, base):
    return np.matmul(np.asarray(digits_array, np.uint), radix_converter(base))

def from_digits(digits_array, base):
    if len(digits_array) > 0:
        if binary_base_test(base):
            return from_bits(digits_array)
        else:
            return _from_digits(digits_array, base)
    else:
        return digits_array




def array_from_string(string_array):
    as_string_array = np.asarray(string_array, dtype=str)
    return np.fromiter(itertools.chain.from_iterable(as_string_array.ravel()), np.uint).reshape(
        as_string_array.shape + (-1,))


def from_string_digits(string_digits_array, base):
    return from_digits(array_from_string(string_digits_array), base)


def _to_digits(integer, base):
    return np.stack(np.unravel_index(integer, base), axis=-1)

def to_digits(integer, base):
    if binary_base_test(base):
        return to_bits(integer,len(base))
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


if __name__ == '__main__':
    integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    base = (2, 3, 2, 3, 4, 2, 2, 3, 2)
    digits_array = to_digits(integers, base)
    print(to_digits(integers, base))
    print(to_string_digits(integers, base))
    print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    integers = 1237
    print(to_digits(integers,base))
    print(from_digits(to_digits(integers,base),base))

    integers = [[234, 1237, 543, 23], [53, 234, 732, 123]]
    base =np.broadcast_to(2,11)
    digits_array = to_digits(integers, base)
    print(to_digits(integers, base))
    print(to_string_digits(integers, base))
    print(np.array_equiv(integers, from_digits(to_digits(integers, base), base)))
    print(np.array_equiv(integers, from_string_digits(to_string_digits(integers, base), base)))
    integers = 1237
    print(to_digits(integers,base))
    print(from_digits(to_digits(integers,base),base))

    print(from_digits([],base))
    print(to_digits([], base))



    def to_bytes_backwards(integers):
        before = np.asarray(integers)
        after = before.view(np.uint8)
        return after.reshape(before.shape+(-1,))

    def to_bits_backwards(integers):
        return np.unpackbits(to_bytes_backwards(integers), axis=-1, bitorder='big')

    def from_bits_backwards(bit_array, dtype):
        # np.packbits(np.flip(bit_array,axis=-1), axis=-1, bitorder='little')
        return np.squeeze(np.packbits(bit_array, axis=-1, bitorder='big').view(dtype),axis=-1)











