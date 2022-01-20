import numpy as np

data_size = 950000
wave_size = 11325
CLASSES_NUM = 256


Y1_MEMMAP_PATH = './memmap/AES_TI_wave_mask_norm_v7_0-500000_960_1110_abdiff_memmap.npy'
Y2_MEMMAP_PATH = './memmap/AES_TI_wave_mask_norm_v7_500000-950000_960_1110_abdiff_memmap.npy'

memmap_y1 = np.memmap(
            filename=Y1_MEMMAP_PATH, dtype=np.float, mode='r',
            shape=(500000, wave_size, 1))
memmap_y2 = np.memmap(
            filename=Y2_MEMMAP_PATH, dtype=np.float, mode='r',
            shape=(450000, wave_size, 1))

X_MEMMAP_PATH = './memmap/AES_TI_wave_mask_norm_v7_0-950000_960_1110_abdiff_memmap.npy'
# Xのデータをmemmapのファイルへ書き込む。

X_memmap = np.memmap(
    filename=X_MEMMAP_PATH, dtype=np.float, mode='w+', shape=(data_size, wave_size, 1))
X_memmap[:500000] = memmap_y1
X_memmap[500000:] = memmap_y2

