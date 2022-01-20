import numpy as np

def shift_rows(data):
    res = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            res[i][j] = data[(i+j)%4][(j) % 4]
    return res

affine_map = np.array([0x58, 0x2D, 0x9E, 0x0B, 0xDC, 0x04, 0x03, 0x24])
affine_map_inv = np.array([0x8C, 0x79, 0x05, 0xEB, 0x12, 0x04, 0x51, 0x53])
iso_map = np.array([0x98, 0xF3, 0xF2, 0x48, 0x09, 0x81, 0xA9, 0xFF])


def new_base(mb, x):  # mb:matrix, x:in
    res = 0
    for i in range(8):
        if x & 1:
            res ^= mb[7-i]
        x >>= 1
    return res


def new_base_array(mb, a):
    res = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            res[i][j] = new_base(mb, a[i][j])
    return res


def affine(a):
    b = new_base(affine_map, a)
    #b ^= 99
    #new_base(iso_map, b)
    return b

def affine_bp(a):
    a_bit = np.zeros(8, dtype=int)
    res = 0
    for i in range(8):
        a_bit[i] = a % 2
        a //= 2

    res += (a_bit[1] ^ a_bit[3]^ a_bit[5]^ a_bit[6])
    res += (a_bit[0] ^ a_bit[1] ^ a_bit[2] ^ a_bit[3]
            ^ a_bit[4] ^ a_bit[5] ^ a_bit[7]) << 1
    res += (a_bit[1] ^ a_bit[2] ^ a_bit[4] ) << 2
    res += (a_bit[0] ^ a_bit[1] ^ a_bit[3] ^ a_bit[5]) << 3
    res += (a_bit[1] ^ a_bit[2] ^ a_bit[6] ) << 4
    res += (a_bit[0] ^ a_bit[1] ^ a_bit[3] ^ a_bit[7]) << 5
    res += (a_bit[0] ^ a_bit[4] ^ a_bit[7]) << 6
    res += (a_bit[1] ^ a_bit[5] ^ a_bit[6] ^ a_bit[7]) << 7
    return res


def mix_columns(data):
    data = new_base_array(iso_map, data)
    res = np.zeros((4, 4), dtype=int)
    for i in range(4):
        a0 = data[0][i]
        a1 = data[1][i]
        a2 = data[2][i]
        a3 = data[3][i]
        b3 = a3 ^ a2
        b2 = a2 ^ a1
        b1 = a1 ^ a0
        b0 = a0 ^ a3
        c0 = a0 ^ a1
        c1 = a2 ^ a3
        v0 = affine_bp(b0)
        v1 = affine_bp(b1)
        v2 = affine_bp(b2)
        v3 = affine_bp(b3)
        res[3][i] = a1 ^ c1 ^ v0
        res[2][i] = c0 ^ a2 ^ v3
        res[1][i] = c0 ^ a3 ^ v2
        res[0][i] = a0 ^ c1 ^ v1
    return res
