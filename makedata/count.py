

import numpy as np
base = "/home/usrs/kenta/work/datasets"
with open(base + '/AES_TI_sakurax/round9_0byte_v1_isomap_0-1000000.csv') as f:
    round_train = f.readlines()

count = np.zeros(256,dtype = 'int')
for i in range(len(round_train)):
    count[int(round_train[i])]+=1

for i in range(256):
    print(count[i])

print("min")
print(count.min())