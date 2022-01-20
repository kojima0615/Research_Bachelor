import random
import numpy as np
import inversion_v2
import state_array



r_con = np.array([
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
])
iso_map = np.array([0x98, 0xF3, 0xF2, 0x48, 0x09, 0x81, 0xA9, 0xFF])
iso_map_inv = np.array([0x64, 0x78, 0x6E, 0x8C, 0x68, 0x29, 0xDE, 0x60])
affine_map = np.array([0x58, 0x2D, 0x9E, 0x0B, 0xDC, 0x04, 0x03, 0x24])
affine_map_inv = np.array([0x8C, 0x79, 0x05, 0xEB, 0x12, 0x04, 0x51, 0x53])

#128bitから4*4行列


def bytes2matrix(k):
    kk = k
    res = []
    for i in range(4):
        res.append([])
    for i in range(4):
        for j in range(4):
            temp = (kk >> 8*(15-4*i-j)) % 256
            res[j].append(temp)
    return np.array(res)

 #4*4行列から128bit


def matrixbytes2(m):
    ml=m.tolist()
    res = 0
    for i in range(4):
        for j in range(4):
            res += (ml[i][j] << (15-j*4-i)*8)
    return res


#和がdata[i][j]になるように分割


def separate_data(data):
    res = np.zeros((4, 4), dtype=int)
    res1 = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            res[i][j] = random.randint(0, data[i][j])
            res1[i][j] = data[i][j] ^ res[i][j]
    return res, res1

def s_box(a):
    out1,out2 = inversion_v2.inverse4(a,0)
    out1 = state_array.affine(out1)
    out1 ^= state_array.affine(out2)
    out1 ^= 0x63
    out1 = new_base(iso_map,out1)
    return out1

def expand_key(key, round):
    #(kround_key)->(k+1round_key)
    res = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            if j == 0:
                if i == 0:
                    res[i][j] = key[0][0] ^ s_box(key[1][3]) ^ new_base(iso_map,r_con[round])
                else:
                    res[i][j] = key[i][0] ^ s_box(key[(i+1) % 4][3])
            else:
                res[i][j] = key[i][j] ^ res[i][j-1]
    return res


def add_roundkey(data, key):
    res = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            res[i][j] = key[i][j] ^ data[i][j]
    return res


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


def check(p1, p2):
    res = np.zeros((4, 4), dtype=int)
    res = new_base_array(iso_map_inv, p1)
    res1 = np.zeros((4, 4), dtype=int)
    res1 = new_base_array(iso_map_inv, p2)
    for i in range(4):
        for j in range(4):
            res[i][j] ^= res1[i][j]
    print(res)


def enc(plain, key):

    #4*4にして二つのデータに分ける
    plain_m1 = bytes2matrix(plain)
    plain_m2 = np.zeros((4,4),dtype=int)

    plain_sbox1 = 0
    #GF(2^8)->GF((2^4)^2)
    plain_m1 = new_base_array(iso_map, plain_m1)
    plain_m2 = new_base_array(iso_map, plain_m2)
    #(上位4bit)*a + (下位4bit) の形 それぞれGF(2^4)

    #鍵もGF(2^4)に
    key_index = new_base_array(iso_map, bytes2matrix(key))

    plain_m1 = add_roundkey(plain_m1, key_index)
    #10round
    for round in range(9):
        #inverse
        for i in range(4):
            for j in range(4):
                out1, out2 = inversion_v2.inverse4(
                    plain_m1[i][j], plain_m2[i][j])
                plain_m1[i][j] = state_array.affine(
                    out1)
                plain_m2[i][j] = state_array.affine(
                    out2)
                plain_m1[i][j] ^= 0x63
        
        if round == 0:
            plain_sbox1 = matrixbytes2(plain_m1) ^ matrixbytes2(plain_m2)
            return plain_sbox1

        #shiftrows 向きを間違えたので二回転置してます
        plain_m1 = state_array.shift_rows(plain_m1.T).T
        plain_m2 = state_array.shift_rows(plain_m2.T).T

        #mixcolumns
        plain_m1 = state_array.mix_columns(plain_m1)
        plain_m2 = state_array.mix_columns(plain_m2)

        key_index = expand_key(key_index, round+1)
        
        plain_m1 = add_roundkey(plain_m1, key_index)

    for i in range(4):
        for j in range(4):
            out1, out2 = inversion_v2.inverse4(
                plain_m1[i][j], plain_m2[i][j])
            plain_m1[i][j] = state_array.affine(
                out1)
            plain_m2[i][j] = state_array.affine(
                out2)
            plain_m1[i][j] ^= 0x63
    #shiftrows
    plain_m1 = state_array.shift_rows(plain_m1.T).T
    plain_m2 = state_array.shift_rows(plain_m2.T).T

    plain_m1 = new_base_array(iso_map, plain_m1)
    plain_m2 = new_base_array(iso_map, plain_m2)

    key_index = expand_key(key_index, 10)

    plain_m1 = add_roundkey(plain_m1, key_index)

    #逆変換
    plain_m1 = new_base_array(iso_map_inv, plain_m1)
    plain_m2 = new_base_array(iso_map_inv, plain_m2)
    res_m = np.zeros((4, 4), dtype=int)
    #一つに
    for i in range(4):
        for j in range(4):
            res_m[i][j] = plain_m1[i][j] ^ plain_m2[i][j]
    #128bitに
    res = matrixbytes2(res_m)
    return res


######################################################
    #流れ

    #1.GF(2^8)->GF((2^4)^2)
    #loop
    #2.addroundkey
    #3.inversion
    #4.12byte目までstate_arrayにつめる
    #5.shiftrows(ソフトウェア実装では全部つめてからシフトしています)
    #6.mixcolumsしながら13~16byteをうまく3つshiftした順につめる
    #loopend
    #7.GF((2^4)^2)->GF(2^8)
#######################################################
