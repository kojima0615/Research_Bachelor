
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import ast
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import Padding1D

def load_sca_model(model_file):
    #model = load_model(model_file, compile = False)
    model = load_model(model_file, compile = False, custom_objects={'Padding1D':Padding1D})
    return model


model = load_sca_model('./512.39.h5')
model.summary()