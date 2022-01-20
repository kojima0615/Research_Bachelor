# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:20:43 2020

@author: kentakojima
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization, Activation, Layer,Dense,GaussianNoise
from tensorflow.keras.layers import add, concatenate

k = tf.keras
kl = tf.keras.layers



class Padding1D(Layer):
    def __init__(self, kind='REFLECT', padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.kind = kind
        super(Padding1D, self).__init__(**kwargs)

    def get_config(self):
        config = {
                'padding': self.padding,
                'kind': self.kind
        }
        base_config = super(Padding1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return tf.pad(input_tensor,  [[0, 0], [padding_left, padding_right], [0, 0]], mode=self.kind)

def get_block(num_kernels, padding_type, activation):
    return [
        Padding1D(padding_type),
        kl.Conv1D(num_kernels, kernel_size=(3), kernel_initializer='he_uniform', use_bias=False),
        kl.BatchNormalization(),
        Activation(activation),
        kl.AveragePooling1D((2))
    ]
def get_block_twov(num_kernels, padding_type, activation,x):
    x = Padding1D(padding_type)(x)
    x = kl.Conv1D(num_kernels, kernel_size=(3), kernel_initializer='he_uniform', use_bias=False)(x)
    x = kl.BatchNormalization()(x)
    x = Activation(activation)(x)
    x = kl.AveragePooling1D((2))(x)
    return x
#v1 32/64/128
#v2 8/16/32
#nagato 64/128/256
# 1/2/4 under
# 2/4/8 under
def scnn_9_elu_symmetric_key(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(128, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_32(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_64_plus(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_32_plus(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_32_noise(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.GaussianNoise(stddev = 0.2),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_8(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(8, 'SYMMETRIC', 'selu'),
        *get_block(8, 'SYMMETRIC', 'selu'),
        *get_block(8, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(64, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(64, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_16(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(16, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(32, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(128, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_64(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_16_cp(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(16, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    y =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    y = kl.concatenate([y,output1])
    output2 = kl.Dense(classes, kernel_initializer='he_uniform', name='output2')(y)
    return Model(input_w,[output1,output2])

def scnn_9_elu_symmetric_16_cp_drop(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(16, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    z = kl.Dense(16, kernel_initializer='he_uniform', name='before_drop')(output1)
    z = kl.Dropout(0.5)(z)
    y =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    y = kl.concatenate([y,z])
    output2 = kl.Dense(classes, kernel_initializer='he_uniform', name='output2')(y)
    return Model(input_w,[output1,output2])



def scnn_9_elu_symmetric_64_noise(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.GaussianNoise(stddev = 0.1),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_64_2(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(512, 'SYMMETRIC', 'selu'),
        *get_block(512, 'SYMMETRIC', 'selu'),
        Padding1D('SYMMETRIC'),
        kl.Conv1D(512, kernel_size=(3), kernel_initializer='he_uniform'),
        kl.BatchNormalization(),
        Activation('selu'),
        #kl.Flatten(),
        kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(512, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def scnn_9_elu_symmetric_32_twov(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(32, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    output2 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output2 = kl.Dense(classes, kernel_initializer='he_uniform', name='output2')(output2)
    return Model(input_w,[output1,output2])

def scnn_9_elu_symmetric_8_twov(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(8, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(8, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(8, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(32, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    output1 =  kl.Dense(32, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    output2 =  kl.Dense(32, kernel_initializer='he_uniform', activation='selu')(x)
    output2 = kl.Dense(classes, kernel_initializer='he_uniform', name='output2')(output2)
    return Model(input_w,[output1,output2])

def scnn_9_elu_symmetric_16_twov(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(16, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(64, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    output1 =  kl.Dense(64, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    output2 =  kl.Dense(64, kernel_initializer='he_uniform', activation='selu')(x)
    output2 = kl.Dense(classes, kernel_initializer='he_uniform', name='output2')(output2)
    return Model(input_w,[output1,output2])

def scnn_9_elu_symmetric_16_twoinput(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    input_B = kl.Input(shape = (256))
    y = kl.Dense(64, kernel_initializer='he_uniform', activation='selu')(input_B)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(64, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    x = kl.concatenate([x, y])
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    return Model([input_w, input_B],output1)

def scnn_9_elu_symmetric_16_twoinput_drop(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    input_B = kl.Input(shape = (256))
    y = kl.Dense(64, kernel_initializer='he_uniform', activation='selu')(input_B)
    y = kl.Dropout(0.5)(y)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(16, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(64, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    x = kl.concatenate([x, y])
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    return Model([input_w, input_B],output1)

def scnn_9_elu_symmetric_32_twoinput(input_size=700, classes=256):
    input_shape = (input_size,1)
    
    input_B = kl.Input(shape = (1))

    y = kl.Dense(8, kernel_initializer='he_uniform', activation='selu')(input_B)
    input_w = kl.Input(shape=input_shape)

    x = get_block_twov(32, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(32, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    x = kl.concatenate([x,y])
    output1 =  kl.Dense(128, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    return Model([input_w, input_B],output1)

def scnn_9_elu_symmetric_64_twoinput(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    input_B = kl.Input(shape = (256))
    #y = kl.Dense(16, kernel_initializer='he_uniform', activation='selu')(input_B)

    x = get_block_twov(64, 'SYMMETRIC', 'selu',input_w)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(64, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(128, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(256, 'SYMMETRIC', 'selu',x)
    x = get_block_twov(256, 'SYMMETRIC', 'selu',x)
    x = Padding1D('SYMMETRIC')(x)
    x = kl.Conv1D(256, kernel_size=(3), kernel_initializer='he_uniform')(x)
    x = kl.BatchNormalization()(x)
    x = Activation('selu')(x)
    x = kl.GlobalMaxPooling1D()(x)
    x = kl.concatenate([x,input_B])
    output1 =  kl.Dense(256, kernel_initializer='he_uniform', activation='selu')(x)
    output1 = kl.Dense(classes, kernel_initializer='he_uniform', name='output1')(output1)
    return Model([input_w, input_B],output1)

def CNN_3(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),

        *get_block(64, 'SYMMETRIC', 'selu'),
        *get_block(128, 'SYMMETRIC', 'selu'),
        *get_block(256, 'SYMMETRIC', 'selu'),
        *get_block(512, 'SYMMETRIC', 'selu'),
        *get_block(512, 'SYMMETRIC', 'selu'),
        #Padding1D('SYMMETRIC'),
        #kl.Conv1D(128, kernel_size=(3), kernel_initializer='he_uniform'),
        #kl.BatchNormalization(),
        #Activation('selu'),
        kl.Flatten(),
        #kl.GlobalMaxPooling1D(),
        #kl.Dropout(0.5),
        kl.Dense(4096, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(4096, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def MLP(input_size=700, classes=256):
    input_shape = (input_size,1)
    model = k.Sequential([
        kl.Input(shape=input_shape),
        kl.Flatten(),
        kl.Dense(128, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(256, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(128, kernel_initializer='he_uniform', activation='selu'),
        kl.Dense(classes, kernel_initializer='he_uniform')
        ])
    return model

def zaid_ASCAD_N50(size):
    
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))
    
    #1st convolutional block
    model.add(kl.Conv1D(32, 1, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))
    #2nd convolutional block
    model.add(kl.Conv1D(64, 25, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(25, strides=25, name = 'block2_pool'))
    #3rd convolutional block
    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block3_pool'))
    model.add(kl.Flatten())
    
    #Classification part
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc2'))
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc3'))
    #model.add(kl.Dropout(0.5))
    #Logits layer
    model.add(kl.Dense(9, name = 'prediction'))
    
    return model

def zaid_ASCAD_N50_key(size):
    
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))
    
    #1st convolutional block
    model.add(kl.Conv1D(32, 1, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))
    #2nd convolutional block
    model.add(kl.Conv1D(64, 25, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(25, strides=25, name = 'block2_pool'))
    #3rd convolutional block
    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block3_pool'))
    model.add(kl.Flatten())
    #model.add(kl.GlobalMaxPooling1D())
    #Classification part
    #20から256にかえた
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc2'))
    model.add(kl.Dense(15, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc3'))
    #model.add(kl.Dropout(0.2))
    #Logits layer
    model.add(kl.Dense(256, name = 'prediction'))
    
    return model

def zaid_ASCAD(size):
	# Designing input layer
    model = k.Sequential()
    model.add(kl.Input(shape=(size,1)))
    # 1st convolutional block
    model.add(kl.Conv1D(8, 3, kernel_initializer='he_uniform', activation='selu', padding='same'))
    model.add(kl.BatchNormalization())
    model.add(kl.AveragePooling1D(2, strides=2))
    model.add(kl.Flatten())
    # Classification layer
    model.add(kl.Dense(20, kernel_initializer='he_uniform', activation='selu'))
    model.add(kl.Dense(20, kernel_initializer='he_uniform', activation='selu'))
    # Logits layer
    model.add(kl.Dense(256))
    return model

def mlp_ASCAD(size):

    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))
    model.add(kl.Flatten())
    for _ in range(5):
        model.add(kl.Dense(200, activation = "relu"))
    model.add(kl.Dense(9,activation = 'softmax'))

    return model


def AES_HD(size):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))
    #1st convolutional block
    model.add(kl.Conv1D(2, 1, kernel_initializer='he_uniform',padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))

    model.add(kl.Flatten())
    model.add(kl.Dense(2, kernel_initializer = 'he_uniform', activation = 'relu', name = 'fc1'))
    model.add(kl.Dense(256, name = 'prediction'))
    
    return model

def CNN_1_key(size):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(64, 11,kernel_initializer='he_uniform',padding='same' , name = 'block1_conv1'))
    #model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))

    model.add(kl.Conv1D(128, 11, kernel_initializer='he_uniform',padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block2_pool'))

    model.add(kl.Conv1D(256, 11, kernel_initializer='he_uniform',padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block3_pool'))

    model.add(kl.Conv1D(512, 11, kernel_initializer='he_uniform',padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(512, 11, kernel_initializer='he_uniform',padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block5_pool'))

    model.add(kl.Flatten())

    model.add(kl.Dense(4096, kernel_initializer = 'he_uniform', activation = 'relu', name = 'fc1'))
    model.add(kl.Dense(4096, kernel_initializer = 'he_uniform', activation = 'relu', name = 'fc2'))
    #Logits layer
    model.add(kl.Dense(256, name = 'prediction'))

    return model


def zaid_ASCAD_N100_key(size):
    
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))
    
    #1st convolutional block
    model.add(kl.Conv1D(32, 1, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))
    #2nd convolutional block
    model.add(kl.Conv1D(64, 50, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(25, strides=25, name = 'block2_pool'))
    #3rd convolutional block
    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('selu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block3_pool'))
    model.add(kl.Flatten())
    
    #Classification part
    #20から256にかえた
    model.add(kl.Dense(20, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dense(20, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc2'))
    model.add(kl.Dense(20, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc3'))
    #model.add(kl.Dropout(0.2))
    #Logits layer
    model.add(kl.Dense(256, name = 'prediction'))
    
    return model

def CNN_2(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(8, 5, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block1_pool'))

    model.add(kl.Conv1D(16, 5, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block2_pool'))
    
    model.add(kl.Conv1D(32, 5, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block3_pool'))
    
    '''
    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=4, name = 'block4_pool'))
    '''
    model.add(kl.Flatten())
    model.add(kl.Dense(128, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model


def mobilenet2(size):
    return k.Sequential([
            kl.Input(shape=(size, 1)),
            kl.Conv1D(64, kernel_size=(3), padding='SAME'),
            kl.BatchNormalization(),
            kl.ReLU(6),
            kl.MaxPooling1D((2)),
            kl.Conv1D(128, kernel_size=(3), padding='SAME'),
            kl.BatchNormalization(),
            kl.ReLU(6),
            kl.MaxPooling1D((2)),
            kl.Conv1D(256, kernel_size=(3), padding='SAME'),
            kl.BatchNormalization(),
            kl.ReLU(6),
            kl.MaxPooling1D((2)),
            kl.Conv1D(256, kernel_size=(3), padding='SAME'),
            kl.BatchNormalization(),
            kl.ReLU(6),
            kl.Conv1D(128, kernel_size=(3), padding='SAME'),
            kl.BatchNormalization(),
            kl.ReLU(6),
            kl.GlobalMaxPooling1D(),
            kl.Dense(128),
            kl.ReLU(6),
            kl.Dense(9, kernel_constraint=k.constraints.unit_norm())
            ])

def mobilenet2_256(size):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling1D(2, strides=2, name = 'block1_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling1D(2, strides=2, name = 'block2_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling1D(2, strides=2, name = 'block3_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling1D(2, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    #model.add(kl.AveragePooling1D(2, strides=2, name = 'block5_pool'))

    model.add(kl.GlobalMaxPooling1D())

    model.add(kl.Dense(128, kernel_initializer = 'he_uniform', activation = 'relu', name = 'fc1'))

    model.add(kl.Dense(256, name = 'prediction'))
    
    return model


def CNN_64(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block1_pool'))
    
    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block2_pool'))
    
    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block3_pool'))
    
    
    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block5_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block6_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block6_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block7_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block7_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block8_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block8_pool'))

    model.add(kl.GlobalMaxPooling1D())
    #model.add(kl.Flatten())
    model.add(kl.Dense(256, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model



def CNN_4(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block1_pool'))
    
    model.add(kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block2_pool'))
    
    model.add(kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block3_pool'))
    
    
    model.add(kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block5_pool'))

    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block6_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block6_pool'))

    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block7_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block7_pool'))

    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block8_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block8_pool'))

    model.add(kl.GlobalMaxPooling1D())
    #model.add(kl.Flatten())
    model.add(kl.Dense(32, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model

def CNN_8(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block1_pool'))
    
    model.add(kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block2_pool'))
    
    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block3_pool'))
    
    
    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block5_pool'))

    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block6_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block6_pool'))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block7_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block7_pool'))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block8_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block8_pool'))

    model.add(kl.GlobalMaxPooling1D())
    #model.add(kl.Flatten())
    model.add(kl.Dense(64, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model

def CNN_16(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block1_pool'))
    
    model.add(kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block2_pool'))
    
    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block3_pool'))
    
    
    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block4_pool'))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block5_pool'))

    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block6_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, strides=2, name = 'block6_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block7_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block7_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block8_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(2, strides=2, name = 'block8_pool'))

    model.add(kl.GlobalMaxPooling1D())
    #model.add(kl.Flatten())
    #model.add(kl.Dropout(0.5))
    model.add(kl.Dense(128, kernel_initializer = 'he_uniform', activation = 'selu', name = 'fc1'))
    #model.add(kl.Dropout(0.5))
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model

def CNN_32(size,classes):
    model = k.Sequential()
    model.add(kl.Input(shape = (size, 1)))

    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block1_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, name = 'block1_pool'))
    
    model.add(kl.Conv1D(32, 3, kernel_initializer='he_uniform', padding='same' , name = 'block2_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, name = 'block2_pool'))
    
    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block3_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, name = 'block3_pool'))
    
    
    model.add(kl.Conv1D(64, 3, kernel_initializer='he_uniform', padding='same' , name = 'block4_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4,  name = 'block4_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block5_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, name = 'block5_pool'))

    model.add(kl.Conv1D(128, 3, kernel_initializer='he_uniform', padding='same' , name = 'block6_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4,  name = 'block6_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block7_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4,  name = 'block7_pool'))

    model.add(kl.Conv1D(256, 3, kernel_initializer='he_uniform', padding='same' , name = 'block8_conv1'))
    model.add(kl.BatchNormalization())
    model.add(kl.Activation('relu'))
    model.add(kl.AveragePooling1D(4, name = 'block8_pool'))

    model.add(kl.GlobalMaxPooling1D())
    #model.add(kl.Flatten())
    #model.add(kl.Dropout(0.5))
    model.add(kl.Dense(256, kernel_initializer = 'he_uniform', activation = 'relu', name = 'fc1'))
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(classes, name = 'prediction'))
    
    return model