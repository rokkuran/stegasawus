import seq

import numpy as np

from os import path
from skimage import io


def bit_generator(s, verbose=False):
    """
    Yields individual bits for characters in string.
    """
    for x in s:
        a = ord(x)
        if verbose:
            print x, ord(x), bin(ord(x))

        i = 0
        while i < 7:
            if verbose:
                print a, bin(a), a & 1
            yield a & 1
            a = a >> 1
            i += 1

    # signify end with 14 zeros (double ascii null)
    for x in xrange(14):
        yield 0


def set_lsb(byte, bit):
    """
    Replaces least significant bit of of byte with bit.
        if bit == 1:
        - 0110101 | 0000001 = 0110101
        - 0110100 | 0000001 = 0110101
        if bit == 0:
        - 0110100 & 1111110 = 0110100
        - 0110101 & 1111110 = 0110100
    """
    if bit:
        return byte | bit
    else:
        return byte & 0b11111110


def embed(I, message, seq_method, verbose=False):
    dimensions = I.shape
    S = I.flatten().copy()
    bits = bit_generator(message)

    pixel_count = 0
    for i in seq_method(n=len(S)):
        bit = next(bits, None)
        if bit is not None:
            S[i] = set_lsb(S[i], bit)
            pixel_count += 1
        else:
            break

    if verbose:
        print 'Pixels modified: %.2f' % (pixel_count / 3.)
    return S.reshape(dimensions)


def reveal(S, seq_method):
    char = ''
    S = S.flatten()

    end = list(np.repeat(1, 14))  # message end signified by 14 zeros
    for i in seq_method(n=len(S)):
        x = S[i]
        bit = x & 1
        char += str(bit)
        end = end[1:] + [bit]
        if not sum(end):
            break

    text = ''
    while len(char) > 0:
        b = char[:7][::-1]  # reversed binary
        text += chr(int(b, 2))
        char = char[7:]

    # remove 2 ascii nulls at message end
    text = text.replace('\x00', '')
    return text


if __name__ == '__main__':
    cdir = path.dirname(__file__)
    I = io.imread(path.join(cdir, '../../data/messages/Lenna.png'))

    def check_embed_reveal(I, msg, seq_method):
        S = embed(I, msg, seq_method)
        return reveal(S, seq_method) == msg

    def test_characters(I, seq_method):
        msg = 'abcdefghijklmnopqrstuvwxys1234567890~`!@#$%^&*()_+-=:<>,.?/|'
        flag = check_embed_reveal(I, msg, seq_method)
        assert flag, 'test_characters: %s' % seq_method.__name__

    # TODO: probably doesn't work for extended ascii codes
    test_characters(I, seq.all_the_kings_men)
    test_characters(I, seq.skipy(y=3))
    test_characters(I, seq.skipy(y=5))
    test_characters(I, seq.skip_rand(seed=0, max_skip=10))
    test_characters(I, seq.skip_rand(seed=77, max_skip=25))
