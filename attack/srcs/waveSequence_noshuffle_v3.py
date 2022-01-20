import math
import os
import subprocess
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence, to_categorical
base = "/home/usrs/kenta/work/datasets/AES_TI_sakurax/memmap"
#X_MEMMAP_PATH = base + '/AES_TI_wave_mask_v5_norm_6000000_down3_memmap.npy'
X_MEMMAP_PATH = base + '/AES_TI_wave_mask_v3_norm_memmap.npy'
#Y_MEMMAP_PATH = base + '/round9_nonshare_hd_v3_isomap_0-4000000_memmap.npy'
#Y_MEMMAP_PATH = base + '/round9_0byte_v3_popcount_isomap_0-4000000_memmap.npy'
Y_MEMMAP_PATH = base + '/round9_0byte_v3_isomap_0-8000000_memmap.npy'
Y1_MEMMAP_PATH = base + '/round9_1byte_v3_isomap_0-4000000_memmap.npy'
#batch_size = 128
CLASSES_NUM = 256
class waveSequence_sh(Sequence):
    """
    学習中の、バッチ単位でのデータの取得を扱うためのクラス。

    Attributes
    ----------
    batch_size : int
        バッチ単体でのデータ件数。
    memmap_X : memmap
        入力データのmemmap配列。
    memmap_y : memmap
        教師データのmemmap配列。
    length : int
        データのインデックス件数。math.ceil(データ行数 / batch_size)の
        値が設定される。

    Parameters
    ----------
    batch_size : int
        バッチ単体でのデータ件数。
    """

    def __init__(self, batch_size, data_start, data_end):
        self.DATA_ROW_NUM = data_end - data_start
        WAVE_SIZE = 1600
        self.batch_size = batch_size
        self.memmap_X = np.memmap(
            filename=X_MEMMAP_PATH, dtype=np.float, mode='r',
            shape=(3000000, WAVE_SIZE, 1))
        #print(self.memmap_X.shape)
        self.memmap_X = self.memmap_X[data_start:data_end]

        self.memmap_y = np.memmap(
            filename=Y_MEMMAP_PATH, dtype=np.int, mode='r',
            shape=(8000000))
        
        self.memmap_y1 = np.memmap(
            filename=Y1_MEMMAP_PATH, dtype=np.int, mode='r',
            shape=(4000000))
        self.memmap_y1 = to_categorical(self.memmap_y1)
        #print(self.memmap_y.shape)
        self.memmap_y = self.memmap_y[data_start:data_end]
        self.memmap_y1 = self.memmap_y1[data_start:data_end]
        self.length = self.DATA_ROW_NUM // batch_size
        print(self.memmap_X.shape)
        print(self.memmap_y.shape)
        print(self.memmap_y1.shape)

    def __getitem__(self, idx):
        """
        対象のインデックスの、バッチ単体分のデータを取得する。

        Parameters
        ----------
        idx : int
            取得対象のインデックス番号。

        Returns
        -------
        X : memmap
            対象のインデックスの入力データ。
        y : memmap
            対象のインデックスの教師データ。
        """
        start_idx = idx * self.batch_size
        last_idx = start_idx + self.batch_size
        X = self.memmap_X[start_idx:last_idx]
        y = self.memmap_y[start_idx:last_idx]
        #y1 = self.memmap_y1[start_idx:min(last_idx,self.DATA_ROW_NUM)]

        return X, y

    def __len__(self):
        """
        データのインデックス件数を取得する。math.ceil(データ行数 / batch_size)の
        値が設定される。

        Returns
        -------
        length : int
            データのインデックス件数。
        """
        return self.length

    def on_epoch_end(self):
        """
        1エポック分の処理が完了した際に実行される。
        属性で持っている（__getitem__関数実行後も残る）データなどの破棄処理や
        コールバックなど、必要な処理があれば記載する。
        """

        # メモリ使用量などを表示する（MB単位）。
        #print(subprocess.getoutput('vmstat -S m'))