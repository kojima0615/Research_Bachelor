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

from rank_256 import rank_fast_256

preds = np.zeros((1,256))
plain = np.zeros(256,dtype = 'uint8')
for i in range(256):
    preds[0][i] = 0.1

out = rank_fast_256(preds,plain)
print(out)
    