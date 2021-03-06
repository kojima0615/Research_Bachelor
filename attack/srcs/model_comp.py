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


#CNN5_4_plainwave_tanh_pool2_sub1_mul3
def model_base(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #??????????????????
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
    #?????????????????????
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

def model_base_mul1(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    #??????????????????
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    y = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #?????????????????????
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

def model_base_mul5(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #??????????????????
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    y = kl.Conv1D(4, 5, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 5, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #?????????????????????
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


def model_base_cutmul(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #??????????????????
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    '''
    y = kl.Conv1D(4, 5, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 5, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    '''
    #?????????????????????
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
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


def model_base_subconversion(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #??????????????????
    '''
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    '''
    #sub_conversion
    x = kl.Conv1D(8, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)

    y = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    
    #?????????????????????
    w = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(x)
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

def model_base_cutsub(input_size=700, classes=256):
    input_shape = (input_size,1)
    input_w = kl.Input(shape=input_shape)
    
    #??????????????????
    '''
    x = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    x = kl.Activation('selu')(x)
    k = kl.Conv1D(4, 1, kernel_initializer='he_uniform', padding='same')(input_w)
    k = kl.Activation('selu')(k)
    x = kl.Subtract()([k, x])
    '''
    y = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(input_w)
    y = kl.Activation('selu')(y)
    z = kl.Conv1D(4, 3, kernel_initializer='he_uniform', padding='same')(input_w)
    z = kl.Activation('tanh')(z)
    w = kl.Multiply()([y,z])
    #?????????????????????
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



