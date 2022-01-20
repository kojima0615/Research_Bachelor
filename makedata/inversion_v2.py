import numpy as np
import inversion2

def bit_inv(a):
    if a==1:
        return 0
    else:
        return 1

def squr4(a):
    a_bit = np.zeros(4, dtype=int)
    a_2bit = np.zeros(2, dtype=int)
    a_bit[0] = a & 1
    a_bit[1] = (a & 2) >> 1
    a_bit[2] = (a & 4) >> 2
    a_bit[3] = (a & 8) >> 3
    a_2bit[0] = a % 4
    a_2bit[1] = a//4
    a_xor = a_2bit[0] ^ a_2bit[1]

    a_xor2 = ((a_xor % 2) << 1) | (a_xor//2)

    b = ((a_bit[1] ^ a_bit[0]) << 1) | a_bit[0]
    out = (a_xor2 << 2) | b
    return out


def gf2_scale_mult(a, b):
    res = np.zeros(2, dtype=int)
    res1 = np.zeros(2, dtype=int)

    #2bitずつに分けるよ
    res[0] = a % 2
    res[1] = a//2
    res1[0] = b % 2
    res1[1] = b//2

    xor_out1 = res[0] ^ res[1]
    xor_out2 = res1[0] ^ res1[1]
    xor_out = bit_inv(xor_out1 & xor_out2)

    mult_out1 = res[1] & res1[1]
    mult_out2 = res[0] & res1[0]

    out = ((mult_out2 ^ xor_out) << 1) | (mult_out2 ^ mult_out1)
    return out


def gf2_mult(a, b):
    res = np.zeros(2, dtype=int)
    res1 = np.zeros(2, dtype=int)

    #2bitずつに分けるよ
    res[0] = a % 2
    res[1] = a//2
    res1[0] = b % 2
    res1[1] = b//2

    xor_out2 = res[0] ^ res[1]
    xor_out1 = res1[0] ^ res1[1]
    xor_out = bit_inv(xor_out1 & xor_out2)

    mult_out1 = bit_inv(res[1] & res1[1])
    mult_out2 = bit_inv(res[0] & res1[0])
    
    out = ((mult_out1 ^ xor_out) << 1) | (mult_out2 ^ xor_out)
    return out


def gf4_mult(a, b):

    res = np.zeros(2, dtype=int)
    res1 = np.zeros(2, dtype=int)

    #2bitずつに分けるよ
    res[0] = a % 4
    res[1] = a//4
    res1[0] = b % 4
    res1[1] = b//4
    scale_out = gf2_scale_mult(res[0] ^ res[1], res1[0] ^ res1[1])
    mult_out1 = gf2_mult(res[1], res1[1])
    mult_out2 = gf2_mult(res[0], res1[0])
    out = ((mult_out1 ^ scale_out) << 2) | (mult_out2 ^ scale_out)
    return out


def gf4_mult_share(d1, d2, d3, d4):
    #maskは後から付けてください今は分からないです

    #fig3をそのまま実装
    res1 = gf4_mult(d1, d3)  
    res2 = gf4_mult(d1, d4)  
    res3 = gf4_mult(d2, d3)  
    res4 = gf4_mult(d2, d4)
    return res1, res2, res3, res4

def gf2_mult_factoring(a, b, f):
    a1 = f
    a0 = ((a & 2) >> 1) ^ (a & 1)
    p2 = bit_inv(a1 & a0)
    p1 = bit_inv(((a & 2) >> 1) & ((b & 2) >> 1))
    p0 = bit_inv(((a & 1)) & ((b & 1)))
    return (p1 ^ p2) << 1 | p2 ^ p0

def gf2_scl_factoring(a,b,f):
    a1 = f
    a0 = ((a&2)>>1)^(a&1)
    p2 = bit_inv(a1&a0)
    p1 = bit_inv(((a & 2) >> 1) & ((b & 2) >> 1))
    p0 = bit_inv(((a & 1)) & ((b & 1)))
    return (p0^p2)<<1 | p1^p0

def gf4_mul_factoring(a, b, ff, f, h, l):

    a_2bit = np.zeros(2, dtype=int)
    b_2bit = np.zeros(2, dtype=int)

    a_2bit[0] = a % 4
    a_2bit[1] = (a & 12) >> 2
    b_2bit[0] = b % 4
    b_2bit[1] = (b & 12) >> 2

    xor_out = a_2bit[0] ^ a_2bit[1]
    out1 = gf2_scl_factoring(xor_out, ff, f)
    out2 = gf2_mult_factoring(a_2bit[1], b_2bit[1], h)
    out3 = gf2_mult_factoring(a_2bit[0], b_2bit[0], l)

    return ((out1 ^ out2) << 2) | (out1 ^ out3)


#(data1,data2) -> (data1+data2)^-1
def inverse4(data1, data2):

    #stage1
    #上位と下位に分割
    data1h = data1//16
    data1l = data1 % 16
    data2h = data2//16
    data2l = data2 % 16
    #highとlowの順番これで合ってるか? あとで確認してください
    '''
    print(data1l,end = ' ')
    print(data2l)
    '''
    s1_1, s1_2, s1_3, s1_4 = gf4_mult_share(data1l, data2l, data1h, data2h)
    '''
    print(data1l, end=' ')
    print(data2l)
    '''
    s1_1 ^= squr4(data1h ^ data1l)
    s1_4 ^= squr4(data2h ^ data2l)
    '''
    print('stage1')
    print(s1_1, end=' ')
    print(s1_2, end=' ')
    print(s1_3, end=' ')
    print(s1_4)
    print(s1_1 ^ s1_2 ^ s1_3 ^ s1_4)
    '''
    #stage2
    #share_gf((2^2)^2)inversion
    s2 = inversion2.inverse2(s1_1 ^ s1_2, s1_3 ^ s1_4)
    s2_1 = 0
    s2_2 = 0

    for i in range(4):
        s2_1 ^= s2[i]

    for i in range(4):
        s2_2 ^= s2[i+4]
    '''
    print('stage2')
    print(s2_1, end=' ')
    print(s2_2)
    print(s2_1 ^ s2_2)
    '''
    #stage3
    #下の変数は暗黒実装でわからん, とりあえず写経した
    ff = np.zeros(2, dtype=int)

    ff[0] = ((s2_1 & 12) >> 2) ^ (s2_1 % 4)
    ff[1] = ((s2_2 & 12) >> 2) ^ (s2_2 % 4)
    f0 = ((ff[0] & 2) >> 1) ^ (ff[0] % 2)
    f1 = ((ff[1] & 2) >> 1) ^ (ff[1] % 2)
    h0 = ((s2_1 & 8) >> 3) ^ ((s2_1 & 4) >> 2)
    h1 = ((s2_2 & 8) >> 3) ^ ((s2_2 & 4) >> 2)
    l0 = ((s2_1 & 2) >> 1) ^ ((s2_1 & 1))
    l1 = ((s2_2 & 2) >> 1) ^ ((s2_2 & 1))
    #print(ff)
    #print(h0)
    #print(h1)
    s3_1 = gf4_mul_factoring(data1l, s2_1, ff[0], f0, h0, l0)
    s3_2 = gf4_mul_factoring(data1l, s2_2, ff[1], f1, h1, l1)
    s3_3 = gf4_mul_factoring(data2l, s2_1, ff[0], f0, h0, l0)
    s3_4 = gf4_mul_factoring(data2l, s2_2, ff[1], f1, h1, l1)
    s3_5 = gf4_mul_factoring(data1h, s2_1, ff[0], f0, h0, l0)
    s3_6 = gf4_mul_factoring(data1h, s2_2, ff[1], f1, h1, l1)
    s3_7 = gf4_mul_factoring(data2h, s2_1, ff[0], f0, h0, l0)
    s3_8 = gf4_mul_factoring(data2h, s2_2, ff[1], f1, h1, l1)

    c0 = s3_1*16 + s3_5
    c1 = s3_2*16 + s3_6
    c2 = s3_3*16 + s3_7
    c3 = s3_4*16 + s3_8
    '''
    print('stage3')
    print(c0, end=' ')
    print(c1, end=' ')
    print(c2, end=' ')
    print(c3)
    print(c1 ^ c2 ^ c3 ^ c0)
    '''
    return c0^c1, c2^c3
