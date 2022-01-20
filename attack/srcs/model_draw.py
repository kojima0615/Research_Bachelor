import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
import cycler
import numpy as np
import sys
from model import * 
from model_v2 import * 
from tensorflow_model_optimization.sparsity import keras as sparsity
from loss import *
import load_data
import shutil
import os
import matplotlib.pyplot as plt
from waveSequence_v2 import *
from waveSequence_noshuffle_v2 import *
from tensorflow.keras.utils import plot_model, to_categorical


s = input()
model: k.Model = eval(s + "({},{})".format(50, 256))
plot_model(model, show_shapes = True,to_file='./image/'+ s + '.png')