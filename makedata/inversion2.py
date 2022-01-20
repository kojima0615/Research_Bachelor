import numpy as np


def bit_inv(a):
    if a == 1:
        return 0
    else:
        return 1


def inverse2(data1, data2):
    a = np.zeros(4, dtype=int)
    b = np.zeros(4, dtype=int)
    res8 = np.zeros(8, dtype=int)
    a[0] = data1 & 1
    a[1] = (data1 & 2) >> 1
    a[2] = (data1 & 4) >> 2
    a[3] = (data1 & 8) >> 3
    b[0] = data2 & 1
    b[1] = (data2 & 2) >> 1
    b[2] = (data2 & 4) >> 2
    b[3] = (data2 & 8) >> 3
    res8[0] += ((a[1] & a[2] | a[0]) ^ a[3]) << 3
    res8[0] += ((a[1] & a[3] | a[0]) ^ a[2]) << 2
    res8[0] += ((a[0] & a[3] | a[2]) ^ a[1]) << 1
    res8[0] += ((a[1] & a[3] | a[2]) ^ a[0])

    res8[1] += ((a[1] & b[2] & bit_inv(a[0])) ^ a[3]) << 3
    res8[1] += ((a[1] & b[3] & bit_inv(a[0])) ^ a[2]) << 2
    res8[1] += ((a[0] & b[3] & bit_inv(a[2])) ^ a[1]) << 1
    res8[1] += ((a[1] & b[3] & bit_inv(b[2])) ^ a[0])

    res8[2] += (b[1] & (bit_inv(a[0] & a[2]) ^ b[3])) << 3
    res8[2] += ((a[0] & b[1] & a[3]) ^ b[1] & a[2] ^ a[0] & a[2]) << 2
    res8[2] += (a[3] & (bit_inv(a[0] & b[2]) ^ a[1])) << 1
    res8[2] += ((a[1] & b[2] & a[3]) ^ a[0] & b[2] ^ a[0] & a[3])

    res8[3] += (a[0]&b[1]&b[2]^b[1]&a[3]) << 3
    res8[3] += ((a[0] & b[1] & b[3]) ^ a[0] & b[2] ^ b[1] & b[2]) << 2
    res8[3] += (b[3]&(a[0]&b[2]^a[1])) << 1
    res8[3] += ((a[1] & a[2] & b[3]) ^ a[0] & a[2] ^ a[0] & b[3])

    for i in range(4):
        temp = a[i]
        a[i] = b[i]
        b[i] = temp
    res8[7] += ((a[1] & a[2] | a[0]) ^ a[3]) << 3
    res8[7] += ((a[1] & a[3] | a[0]) ^ a[2]) << 2
    res8[7] += ((a[0] & a[3] | a[2]) ^ a[1]) << 1
    res8[7] += ((a[1] & a[3] | a[2]) ^ a[0])

    res8[6] += ((a[1] & b[2] & bit_inv(a[0])) ^ a[3]) << 3
    res8[6] += ((a[1] & b[3] & bit_inv(a[0])) ^ a[2]) << 2
    res8[6] += ((a[0] & b[3] & bit_inv(a[2])) ^ a[1]) << 1
    res8[6] += ((a[1] & b[3] & bit_inv(b[2])) ^ a[0])

    res8[5] += (b[1] & (bit_inv(a[0] & a[2]) ^ b[3])) << 3
    res8[5] += ((a[0] & b[1] & a[3]) ^ b[1] & a[2] ^ a[0] & a[2]) << 2
    res8[5] += (a[3] & (bit_inv(a[0] & b[2]) ^ a[1])) << 1
    res8[5] += ((a[1] & b[2] & a[3]) ^ a[0] & b[2] ^ a[0] & a[3])

    res8[4] += (a[0] & b[1] & b[2] ^ b[1] & a[3]) << 3
    res8[4] += ((a[0] & b[1] & b[3]) ^ a[0] & b[2] ^ b[1] & b[2]) << 2
    res8[4] += (b[3] & (a[0] & b[2] ^ a[1])) << 1
    res8[4] += ((a[1] & a[2] & b[3]) ^ a[0] & a[2] ^ a[0] & b[3])
    return res8
