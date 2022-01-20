import encryption_round
import csv
import numpy as np
iso_map_inv_old = np.array([0x64, 0x78, 0x6E, 0x8C, 0x68, 0x29, 0xDE, 0x60])
iso_map = np.array([0x6C, 0x42, 0xED, 0xEB, 0x12, 0x04, 0x26, 0x94])
iso_map_inv = np.array([0x39, 0x74, 0x32, 0x3C, 0xC2, 0x04, 0x34, 0x99])
def new_base(mb, x):  # mb:matrix, x:in
    res = 0
    for i in range(8):
        if x & 1:
            res ^= mb[7-i]
        x >>= 1
    return res


with open("./csv_data/plain_4000000_v2.csv") as f:
    plain_text = f.readlines()

with open("./csv_data/key_4000000_v2.csv") as f:
    key = f.readlines()

out = [0]*4000000
out1 = [0]*4000000
for i in range(4000000):
    out[i], out1[i] = encryption_round.enc(int(plain_text[i]),int(key[i]))
    out[i] =new_base(iso_map_inv_old,out[i])
    out[i] = new_base(iso_map,out[i])
    out1[i] = new_base(iso_map,out1[i])
    '''
    if i % 1000 == 0:
        print(i)
    '''



with open("./csv_data/round10_1byte_key_v3_isomap_0-4000000.csv", "w") as f:
    for ele in out:
        f.write(str(ele)+'\n')

with open("./csv_data/round10_1byte_v3_isomap_0-4000000.csv", "w") as f:
    for ele in out1:
        f.write(str(ele)+'\n')