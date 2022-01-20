# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:16:11 2020

@author: kentakojima
"""
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import ast
import tensorflow as tf
from tensorflow.keras.models import load_model
import pdb 
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
from tensorflow.keras import backend as K
from multiprocessing import Pool
from scipy.special import softmax
from loss import *
from scipy.stats import binom
from argparse import ArgumentParser
from distutils.util import strtobool
import math
from joblib import Parallel, delayed
import load_data
from model import Padding1D
from waveSequence_predict import *
from numba import jit
np.random.seed(seed=34)
base ="/home/usrs/kenta/work/datasets/AES_TI_sakurax" 
base_global = "/home/usrs/kenta"
with open(base_global + '/AES_TI_FPGA_getdata/sboxout.txt') as f:
    sboxout = f.readlines()

dict_sbox = np.zeros(pow(2,16),dtype = int)

for i in range(len(sboxout)):  
    dict_sbox[int('0x'+sboxout[i],16)] = i

hw = np.array([bin(i).count('1') for i in range(256)])
inv = np.load(base + '/csv_data/inv_LUT.npy')
pdf = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1],dtype = np.float)
pdf_16 = np.array([1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870,11440,8008,4368,1820,560,120,16,1  ],dtype = np.float)
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
inv_s_box = (
        0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
        0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
        0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
        0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
        0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
        0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
        0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
        0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
        0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
        0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
        0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
        0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
        0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
        0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
        0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
        0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
    )

np.set_printoptions(threshold=9999999999)

argparser = ArgumentParser()

argparser.add_argument('model', type = str, default = None)
#ここからはオプション引数
argparser.add_argument('-b', '--binom', type = int, default = True)
argparser.add_argument('-o', '--output', type = str, default = None)
argparser.add_argument('-a', '--average', type = int, default = 1)
argparser.add_argument('-n', '--num-traces', type = int, default = 2000)
argparser.add_argument('-d', '--dataset', type = str, default = "AES_RD") 

args = argparser.parse_args()


def load_sca_model(model_file):
    #model = load_model(model_file, compile = False)
    model = load_model(model_file, compile = False, custom_objects={'Padding1D':Padding1D})
    return model

def model_fit(model):
    predict_gen = waveSequence_pred(128, 0, 500000)
    #base_np = "/home/usrs/kenta/work/datasets/AES_TI_sakurax/trace"
    #input_data = np.load(base_np + "/AES_TI_wave_mask_v4_test_norm_down3.npy")
    ##input_data = input_data.reshape(-1,1600,1)
    #推測値を正規化
    predictions = softmax(model.predict_generator(predict_gen), axis =1)
    #predictions = softmax(model.predict(input_data), axis =1)
    #del input_data
    print(predictions.shape)
    return predictions

iso_map = np.array([0x6C, 0x42, 0xED, 0xEB, 0x12, 0x04, 0x26, 0x94])
iso_map_inv = np.array([0x39, 0x74, 0x32, 0x3C, 0xC2, 0x04, 0x34, 0x99])

@jit(nopython=True)
def new_base(mb, x):  # mb:matrix, x:in
    res = 0
    for i in range(8):
        if x & 1:
            res ^= mb[7-i]
        x >>= 1
    return res

def hd_TI(key, pt):
    res = pt
    res ^= key
    
    res = new_base(iso_map_inv,res)
    res = inv_s_box[res]
    res = new_base(iso_map,res)
    
    return res
    

@jit(nopython=True)
def rank_key(X,pt):
    likelihood = np.zeros((X.shape[0],256))
    for i in range(X.shape[0]):
        for key in range(256):
            #引数を取るときにindexでしていしているのでpt[i][0]
            #outs = Sbox[key ^ pt[i]]
            #round9にpt[i]をいれたときのHD
            out = pt[i] ^ key
            out = new_base(iso_map_inv,out)
            out = inv_s_box[out]
            out = new_base(iso_map,out)

            '''
            #9class
            out = hd[out]
            likelihood[i][key] += np.log(X[i][out]) / pdf[out]
            '''

            likelihood[i][key] += np.log(X[i][out])
            if i!=0:
                likelihood[i][key] += likelihood[i-1][key]
    for i in range(X.shape[0]):
        likelihood[i] /= i+1
    return likelihood
    
    

def loop(i,preds,plaintext, key, index):
    indices = np.arange(preds.shape[0])
    
    if i > 0:
        indices = np.random.permutation(indices)
    indices = indices[:num_traces]
    X, pt = preds[indices], plaintext[indices]
    
    #from rank import rank_fast
    
    #hwのとき
    #out = rank_fast(X, pt[:,index].ravel(), args.binom)
    #keyのとき
    '''
    out = rank_key(X, pt[:,index].ravel())
    out = 255 - np.argsort(np.argsort(out))[:,key[index]].ravel()
    '''
    #0byteしかloadしてないとき
    out = rank_key(X, pt)
    #高速化バージョン
    #from rank_256 import rank_fast_256
    #from rank_256 import rank_fast_256
    #out = rank_fast_256(X, pt)
    '''
    plt.plot(out)
    plt.grid()
    plt.show()
    '''
    out = 255 - np.argsort(np.argsort(out))[:,key].ravel()
    del X
    del pt
    del indices
    return np.array(out)

@jit(nopython=True)
def rank_share(X, pt_s1,pt_s2,r_s1,r_s2):

    likelihood = np.zeros((X.shape[0],256))
    for i in range(X.shape[0]):
        for key in range(256):
            out_t = key ^ pt_s1[i]
            out_t = (out_t << 8) + pt_s2[i]  
            out_t = dict_sbox[out_t]
            out = hw[(out_t%256) ^ r_s2[i]] + hw[(out_t//256) ^ r_s1[i]]
            likelihood[i][key] += np.log(X[i][out]) #/ pdf[out]
            if i!=0:
                likelihood[i][key] += likelihood[i-1][key]
    for i in range(X.shape[0]):
        likelihood[i] /= i+1
    return likelihood

def loop_share(i,preds,round9_s1,round9_s2,plaintext_sh1,plaintext_sh2):
    indices = np.arange(preds.shape[0])
    
    if i > 0:
        indices = np.random.permutation(indices)
    indices = indices[:num_traces]

    X, pt_s1,pt_s2,r_s1,r_s2 = preds[indices], plaintext_sh1[indices], plaintext_sh2[indices],\
    round9_s1[indices], round9_s2[indices]
    
    out = rank_share(X, pt_s1,pt_s2,r_s1,r_s2)
    out = 255 - np.argsort(np.argsort(out))[:,89].ravel()
    
    del X
    del pt_s1,pt_s2,r_s1,r_s2
    del indices
    
    return np.array(out)

@jit(nopython=True)
def rank_nonshare(X, pt, r):

    likelihood = np.zeros((X.shape[0],256))
    for i in range(X.shape[0]):
        for key in range(256):
            #iso
            '''
            out = pt[i] ^ key
            out = new_base(iso_map_inv,out)
            out = Sbox[out]
            out = new_base(iso_map,out)
            out = out ^ r[i]
            '''

            #noniso
            out = pt[i] ^ key
            out = inv[out]
            #out = hw[out ^ r[i]]
            out = out ^ r[i]
            likelihood[i][key] += np.log(X[i][out]) #- np.log(pdf[out])
            if i!=0:
                likelihood[i][key] += likelihood[i-1][key]
    for i in range(X.shape[0]):
        likelihood[i] /= i+1
    return likelihood

def loop_nonshare(i,preds,R,PT,key):
    indices = np.arange(preds.shape[0])
    
    if i > 0:
        indices = np.random.permutation(indices)
    indices = indices[:num_traces]

    X, pt, r = preds[indices], PT[indices], R[indices]
    out = rank_nonshare(X, pt, r)
    out = 255 - np.argsort(np.argsort(out))[:,key].ravel()
    
    del X
    del pt, r
    del indices
    
    return np.array(out)

@jit(nopython=True)
def rank_twov(X, pt1, pt5):
    #nonisomap    
    likelihood = np.zeros((X.shape[0],256 * 256))
    for i in range(X.shape[0]):
        for key1 in range(256):
            for key5 in range(256):
                key15 = key1 + (key5 << 8)
                out = inv_s_box[pt1[i] ^ key1] ^ inv_s_box[pt5[i] ^ key5]
                #out = new_base(iso_map, out)
                likelihood[i][key15] += np.log(X[i][out])
                if i!=0:
                    likelihood[i][key15] += likelihood[i-1][key15]
    for i in range(X.shape[0]):
        likelihood[i] /= i+1
    return likelihood

def loop_twov(i, preds, PT1, PT5, key1, key5):
    indices = np.arange(preds.shape[0])
    
    if i > 0:
        indices = np.random.permutation(indices)
    indices = indices[:num_traces]

    X, pt1, pt5 = preds[indices], PT1[indices], PT5[indices]
    out = rank_twov(X, pt1, pt5)
    out = (256*256 - 1) - np.argsort(np.argsort(out))[:,key1 + (key5 << 8)].ravel()
    
    del X
    del pt1, pt5
    del indices
    return np.array(out)

def add_prefix(name, pre):
    n = name.split('.')[:-1]
    p = name.split('.')[-1]
    return '.'.join(n) + "_" + pre + "." + p


if __name__=="__main__":
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    dataset = args.dataset
    model_file = args.model #mod_fileから持ってきたやつ
    fig_name = args.output
    num_traces = args.num_traces
    
    metadata = load_data.metadata(dataset)
    plaintext = metadata["plaintext"]
    key = metadata["key"]
    
    ge = np.zeros(num_traces) #guessing entropy
    sc = np.zeros(num_traces) #success rate
    
    print("loading model")
    model = load_sca_model(model_file)
    print("model_fit")
    preds = model_fit(model)
    del model
    print(preds.shape)
    print("rank calculation")
    index = 0
    
    if "ASCAD" in dataset:
        index = 2
        #ASCADは3byte目をみるから？
    #outs = Parallel(n_jobs = 16, verbose=3)([delayed(loop)(i, preds, plaintext, key, index)for i in range(args.average)])
    '''
    #share(CPA)
    outs = Parallel(n_jobs = 16, verbose=3)([delayed(loop_share)(i, preds, \
        round9_1byte_s1, round9_1byte_s2, round10_0byte_s1, round10_0byte_s2)for i in range(args.average)])
    
    '''
    
    r = np.load(base + '/csv_data/round1_inv_5byte_test_isomap_4000000_v3.npy')

    plaintext5 = np.load(base + '/csv_data/round1_plain_5byte_test_isomap_4000000_v3.npy')
    key5 = np.load(base + '/csv_data/round1_key_5byte_test_isomap_4000000_v3.npy')[0]
    
    outs = Parallel(n_jobs = 16, verbose=3)([delayed(loop_nonshare)(i, preds, \
        r, plaintext, key)for i in range(args.average)])
    
    
    #nonisomap_twovalue
    '''
    outs = Parallel(n_jobs = 2, verbose=3)([delayed(loop_twov)(i, preds, \
        plaintext, plaintext5, key, 227)for i in range(args.average)])
    '''

    #n_jobs 利用するコア数
    #verbose 進捗を出力する頻度
    for i in range(args.average):
        ge += outs[i]
        sc += (outs[i]==0)
    ge /=args.average
    sc /= args.average
    '''
    plt.plot(sc)
    plt.show()
    '''
    np.save(add_prefix(fig_name, "GE"), ge)
    np.save(add_prefix(fig_name, "SC"), sc)
    
    
    
    
    
    