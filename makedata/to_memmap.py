import numpy as np

data_size = 1500000
#wave_size = 9730
wave_size = 500
CLASSES_NUM = 256

trace = np.load("./trace/AES_TI_wave_mask_log_v8_0-1500000_1200_1700.npy")
trace = trace.reshape(data_size,-1,1)
wave_size = trace.shape[1]
print(trace.shape)

hw = np.array([bin(i).count('1') for i in range(256)])
'''
with open('../AES_TI_sakurax/csv_data/round10_before_ak_test_1byte_v3_isomap_0-1000000.csv') as f:
    round_train_0 = f.readlines()
with open('../AES_TI_sakurax/csv_data/round10_before_ak_1byte_v3_isomap_0-4000000.csv') as f:
    round_train_1 = f.readlines()
round_train_0 = np.array(round_train_0, dtype=np.int)
round_train_1 = np.array(round_train_1, dtype=np.int)

#round_train_0 = np.concatenate([round_train_0, round_train_0])
#round_train_1 = np.concatenate([round_train_1, round_train_1])
print(round_train_0.shape)
round_train_0 = round_train_0[:data_size]
round_train_1 = round_train_1[:data_size]

round_train_2 = round_train_0 ^ round_train_1
#round_train_2 = hw[round_train_0 ^ round_train_1]
'''
round_train_0 = np.load('./csv_data/round1_inv_1byte_isomap_4000000_v3.npy')
round_train_1 = np.load('./csv_data/round1_inv_5byte_isomap_4000000_v3.npy')
# memmap用のファイルのパス。
X_MEMMAP_PATH = './memmap/AES_TI_wave_mask_log_v8_0-1500000_1200_1700_memmap.npy'

Y_MEMMAP_PATH = './memmap/round1_inv_15byte_xor_isomap_4000000_v3_memmap.npy'
Y1_MEMMAP_PATH = './memmap/round1_inv_15byte_hd_isomap_4000000_v3_memmap.npy'
Y2_MEMMAP_PATH = './memmap/round1_inv_5byte_isomap_4000000_v3_memmap.npy'

# Xのデータをmemmapのファイルへ書き込む。

X_memmap = np.memmap(
    filename=X_MEMMAP_PATH, dtype=np.float, mode='w+', shape=(data_size, wave_size, 1))
X_memmap[:] = trace


# yのデータをmemmapのファイルへ書き込む。
'''
y_memmap = np.memmap(
    filename=Y_MEMMAP_PATH, dtype=np.int, mode='w+', shape=(data_size))
y_memmap[:] = round_train_0 ^ round_train_1

y1_memmap = np.memmap(
    filename=Y1_MEMMAP_PATH, dtype=np.int, mode='w+', shape=(data_size))
y1_memmap[:] = hw[round_train_1 ^ round_train_0]
'''
'''
y2_memmap = np.memmap(
    filename=Y2_MEMMAP_PATH, dtype=np.int, mode='w+', shape=(data_size))
y2_memmap[:] = round_train_2
'''
