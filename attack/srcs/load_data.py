# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:01:37 2020

@author: kentakojima
"""
import numpy as np
#import h5py
#from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from scipy import signal

Sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
)
iso_map = np.array([0x6C, 0x42, 0xED, 0xEB, 0x12, 0x04, 0x26, 0x94])
iso_map_inv = np.array([0x39, 0x74, 0x32, 0x3C, 0xC2, 0x04, 0x34, 0x99])
def new_base(mb, x):  # mb:matrix, x:in
    res = 0
    for i in range(8):
        if x & 1:
            res ^= mb[7-i]
        x >>= 1
    return res
#3byte目を返して
def trans(data):
    res = []
    for i in range(16):
        res.append((data>>8*(15-i))%256)
    return res


#二次元配列に対して
def normalize(data):
    data = data.astype(np.float)
    data -= data.mean()
    data /= data.std()
    #data = np.square(data)
    return data

def to_hw(data):
    data = np.array([bin(x).count("1") for x in data])
    return data

def to_hw_str(data):
    data = np.array([bin(int(x)).count("1") for x in data])
    return data

def normalize_mat(data):
    return np.array([normalize(v) for v in data])

def mk_label_ASCAD(f, mask = False):
    #1round, 3byte目のSboxの出力で比較
    p = f['plaintext']
    k = f['key']
    m = f['masks']
    
    if mask is False:
        return np.array([Sbox[p[i][2] ^ k[i][2]] ^ m[i][2] for i in range(len(p))]).astype(np.uint8)
    else:
        return np.array([Sbox[p[i][2] ^ k[i][2]] for i in range(len(p))]).astype(np.uint8)
    
def mk_label_sakurax(round1):
    res = np.zeros(len(round1))
    for i in range(len(round1)):
        res[i] = int(round1[i])
    return res.astype(np.uint8)

def mk_label_sakurax_twov(round1,round2):
    res = np.zeros(len(round1))
    for i in range(len(round1)):
        res[i] = int(round1[i]) ^ int(round2[i])
    return res.astype(np.uint8)


def data_flat(data, label):
    # 全クラスのサンプル数
    sample_nums = np.array([])

    for i in range(max(label)+1):
    # 各クラスのサンプル数
        sample_num = np.sum(label == i)
        # サンプル数管理配列に追加
        sample_nums = np.append(sample_nums, sample_num)

    # 全クラス内の最小サンプル数
    min_num = np.min(sample_nums)
    del_indexes = []
    for i in range(len(sample_nums)):
        # 対象クラスのサンプル数と最小サンプル数の差
        diff_num = int(sample_nums[i] - min_num)

        # 削除する必要が無い場合はスキップ
        if diff_num == 0:
            continue

        # 削除する要素のインデックス
        # タプルになっているのでlistに変換 (0番目のインデックスに配置されている)
        indexes = list(np.where(label == i)[0])

        # 削除するデータのインデックス
        del_indexes.extend(random.sample(indexes, diff_num))

        # データから削除
    print("d_data")
    data = np.delete(data, del_indexes,0)
    print("d_label")
    label = np.delete(label, del_indexes)
    print(len(del_indexes))
    print(data.shape)
    return data, label

def traces(dataset,model_name):
    base = "/home/usrs/kenta/work/datasets"
    if dataset == "AES_RD":
        train_traces = normalize_mat(np.load(base + "/AES_RD_profiling/profiling_traces_AES_RD.npy"))
        train_labels = to_hw(np.load(base + "/AES_RD_profiling/profiling_labels_AES_RD.npy"))

        test_traces = normalize_mat(np.load(base + "/AES_RD_attack/attack_traces_AES_RD.npy"))
        test_labels = to_hw(np.load(base + "/AES_RD_attack/attack_labels_AES_RD.npy"))
        train_traces = train_traces.reshape(-1, train_traces.shape[1], 1)
        test_traces = test_traces.reshape(-1, train_traces.shape[1], 1)
    elif "ASCAD" in dataset:
        f = h5py.File('../datasets/ASCAD_fixed/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r')
        train_traces = normalize_mat(np.array(f['Profiling_traces']['traces']).astype(np.float))
        test_traces = normalize_mat(np.array(f['Attack_traces']['traces']).astype(np.float))
        train_traces = train_traces.reshape(-1, 700, 1)
        test_traces = test_traces.reshape(-1, 700, 1)
        #正解ラベルはハミングウェイト
        if "key" in model_name:
            if "with_mask" in dataset:
                train_labels = mk_label_ASCAD(f['Profiling_traces']['metadata'], True)
                test_labels = mk_label_ASCAD(f['Attack_traces']['metadata'], True)
            else:
                train_labels = mk_label_ASCAD(f['Profiling_traces']['metadata'], False)
                test_labels = mk_label_ASCAD(f['Attack_traces']['metadata'], False)
        else:
            if "with_mask" in dataset:
                train_labels = to_hw(mk_label_ASCAD(f['Profiling_traces']['metadata'], True))
                test_labels = to_hw(mk_label_ASCAD(f['Attack_traces']['metadata'], True))
            else:
                train_labels = to_hw(mk_label_ASCAD(f['Profiling_traces']['metadata'], False))
                test_labels = to_hw(mk_label_ASCAD(f['Attack_traces']['metadata'], False))

    elif "AES_TI_sakurax" in dataset:
        wave_size = 1600
        #train_traces_0 = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_1000000_sakurax_10round', np.uint8)
        #train_traces_0 = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_1000000_sakurax_10round_unmask', np.uint8)
        #train_traces_0 = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_0-1000000', np.uint8)
        
        train_traces_0 = np.load(base + "/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_0-1000000_down3.npy")
        #train_traces_0 = np.load(base + "/AES_TI_sakurax/trace/AES_TI_wave_1000000_sakurax_10round_unmask_down2.npy")
        train_traces_0 = train_traces_0.reshape(-1, wave_size)
        train_traces=train_traces_0
        del train_traces_0
        print("t_0")
        
        #train_traces_1 = np.fromfile(base + '/AES_TI_sakurax/AES_TI_wave_mask_v3_1000000-2000000', np.uint8)
        train_traces_1 = np.load(base + "/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_1000000-2000000_down3.npy")
        train_traces_1 = train_traces_1.reshape(-1, wave_size)
        train_traces = np.concatenate([train_traces, train_traces_1])
        del train_traces_1
        print("t_1")
    
        #train_traces_2 = np.fromfile(base + '/AES_TI_sakurax/AES_TI_wave_mask_v3_2000000-3000000', np.uint8)
        train_traces_2 = np.load(base + "/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_2000000-3000000_down3.npy")
        train_traces_2 = train_traces_2.reshape(-1, wave_size)
        train_traces = np.concatenate([train_traces, train_traces_2])
        del train_traces_2
        print("input trace")
        
        #unmasek:4880
        train_size = 3000000
        #train_size = 1
        wavel = 0
        waver = wave_size
        print(train_traces.shape)
        train_traces = train_traces[:train_size,wavel:waver]

        #時間ずれしたやつをくっつける
        #train_traces_out=np.concatenate([train_traces[:train_size-100000, wavel-50:waver-50] ,train_traces_out])
        #train_traces_out=np.concatenate([train_traces[:train_size-100000, wavel:waver] + np.random.normal(0.0,0.2,(train_size-100000,waver-wavel)),train_traces_out])
        #test_traces = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_test_400000_sakurax_10round', np.uint8)
        test_traces = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_test_400000_sakurax_10round_unmask', np.uint8)
        #test_traces = np.load(base + "/AES_TI_sakurax/trace/AES_TI_wave_test_400000_sakurax_10round_unmask_down2.npy")
        #test_traces = np.fromfile(base + '/AES_TI_sakurax/AES_TI_wave_mask_v3_test', np.uint8)
        #test_traces = np.load(base + '/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_test_down3.npy')
        test_traces = test_traces.reshape(-1, wave_size)
        #test_size = 400000
        test_size = 1
        print(train_traces.shape)
        test_traces = test_traces[:test_size,wavel:waver]

        
        train_traces = normalize_mat(train_traces)
        test_traces = normalize_mat(test_traces)
        print("train_labels")
        #train_labels
        '''
        with open(base + '/AES_TI_sakurax/csv_data/round9_0byte_v3_isomap_0-4000000.csv') as f:
            round_train_0 = f.readlines()
        '''
        
        with open(base + '/AES_TI_sakurax/csv_data/round10_before_ak_0byte_v3_isomap_0-4000000.csv') as f:
            round_train_0 = f.readlines()
        with open(base + '/AES_TI_sakurax/csv_data/round10_before_ak_1byte_v3_isomap_0-4000000.csv') as f:
            round_train_1 = f.readlines()
        
        #test_labels metadataから取ってくるのでイラン
        train_labels_0 = mk_label_sakurax(round_train_0)
        train_labels_1 = mk_label_sakurax(round_train_1)
        #train_labels = mk_label_sakurax_twov(round_train_0,round_train_1)
        #train_labels = to_hw_str(round_train)

        #データセットがバカでかいので適当に切ってください
        #波形も適当な場所でスライスしてください
        train_labels_0 = train_labels_0[:train_size]
        train_labels_1 = train_labels_1[:train_size]
        
        print("flat")
        
        train_traces = train_traces.reshape(-1,train_traces.shape[1],1)
        test_traces = test_traces.reshape(-1, test_traces.shape[1], 1)
        

        plt.plot(train_traces[0])
        plt.ylim(ymin=-10,ymax=10)
        plt.grid(True)
        plt.legend()
        plt.savefig("./image/sakurax_test_image")
        print(train_labels_0.shape)
        print(train_labels_1.shape)
        train_labels_0 += (train_labels_1 << 8)
        #return (train_traces, train_labels), (test_traces, train_labels)
        return (train_traces, train_labels_0), (test_traces, train_labels_1)

    #print(train_traces)
    plt.plot(train_traces[0])
    plt.ylim(ymin=-2.5,ymax=2.5)
    plt.grid(True)
    plt.legend()
    plt.savefig("./image/ascad_test_image")
    print("shape")
    print(train_traces.shape)
    print(train_labels.shape)
    print("type")
    print(train_traces.dtype)
    print(train_labels.dtype)
    return (train_traces, train_labels), (test_traces, test_labels)



def metadata(dataset):
    base = "/home/usrs/kenta/work/datasets"
    
    res = {}
    if dataset == "AES_RD":
        res["plaintext"] = np.load(base + "/AES_RD_attack/attack_plaintext_AES_RD.npy")
        res["key"] = np.load(base + "/AES_RD_attack/key.npy")
    elif "ASCAD" in dataset:
        f = h5py.File('../datasets/ASCAD_fixed/ASCAD_data/ASCAD_databases/ASCAD.h5', 'r')
        res["plaintext"] = np.array(f['Attack_traces']['metadata']['plaintext']).astype(np.uint8)
        res["key"] = np.array(f['Attack_traces']['metadata']['key'])[0].astype(np.uint8)
        res["mask"] = np.array(f['Attack_traces']['metadata']['masks'])[0].astype(np.uint8)
        #keyとmaskは固定
    
    elif ("AES_TI" in dataset) or ("AES_nonTI" in dataset):
        '''
        with open(base + '/AES_TI_sakurax/csv_data/round10_test_0byte_v3_nonisomap_0-4000000.csv') as f:
            pt = f.readlines()
        
        with open(base + '/AES_TI_sakurax/csv_data/round10_test_0byte_key_v3_nonisomap_0-4000000.csv') as f:
            key = f.readlines()
        '''
        pt = np.load(base + '/AES_TI_sakurax/csv_data/round1_plain_1byte_test_isomap_4000000_v3.npy')
        key = np.load(base + '/AES_TI_sakurax/csv_data/round1_key_1byte_test_isomap_4000000_v3.npy')
        
        res["plaintext"] = np.array(pt).astype(np.int)
        res["key"] = np.array(key)[0].astype(np.int)

    return res
    
    
    
        
        
        