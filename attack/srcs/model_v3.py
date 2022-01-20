import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Subtract,Multiply ,Dense, Input, \
    RepeatVector, Conv1D, AveragePooling1D, MaxPooling1D,  BatchNormalization, Activation, Layer,Dense,GaussianNoise
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



class MyBias(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyBias, self).__init__(**kwargs)

    def build(self):
        self.bias  = self.add_weight(name='bias',\
                                    shape=(1, self.output_dim), initializer='he_uniform')
        super(MyBias, self).build()

    def call(self):
        return self.bias

    def compute_output_shape(self):
        return(self.output_dim)


def CNN2_16_plainwave(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('selu')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)

def CNN2_4_plainwave(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('selu')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)


def CNN2_8_plainwave_tanh_cutsub(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    '''
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    '''
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)



def CNN2_8_plainwave_tanh_8_8(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    x = kl.Conv1D(8, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x_1 = kl.Conv1D(8, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x_1 = kl.Activation('selu')(x_1)
    x = kl.Subtract()([x_1, x])
    y = kl.Conv1D(8, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(8, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)


    
def CNN2_8_plainwave_tanh_pool4(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)



def CNN3_4_plainwave_tanh_pool4(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)

    
def CNN3_8_plainwave_tanh_pool4(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(4, strides=4)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)



def CNN8_4_plainwave_tanh_pool2(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(16, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)


def CNN5_8_plainwave_tanh_pool2(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    x = kl.Conv1D(1, 1, kernel_initializer='he_uniform', padding='same')(x)
    x = kl.Activation('selu')(x)
    x = kl.Subtract()([input_w, x])
    
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)


def CNN5_4_plainwave_tanh_pool2_sub1_mul3(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    y = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)


def CNN5_4_plainwave_tanh_pool2_sub2_mul2(input_size=700, classes=256, classes_1 = 256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #前処理を模倣
    
    x = kl.Conv1D(2, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(2, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    
    y = kl.Conv1D(2, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(2, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #ここまで前処理
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Conv1D(8, 3, kernel_initializer='he_uniform', padding='same')(w)
    w = kl.BatchNormalization()(w)
    w = kl.Activation('selu')(w)
    w = kl.AveragePooling1D(2, strides=2)(w)
    w = kl.Flatten()(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    w = kl.Dense(20, kernel_initializer='he_uniform', activation='selu')(w)
    output = kl.Dense(classes)(w)
    return Model(input_w,output)
