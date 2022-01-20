# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:04:25 2020

@author: kentakojima
"""

import tensorflow as tf
from scipy.stats import binom
import numpy as np

def var_met(y_true, y_pred):
    HW = np.arange(9)
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.nn.softmax(y_pred, axis=1)
    y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    var = tf.reduce_sum(y_pred * HW * HW, axis=1)
    middle = -2 * y_true * tf.reduce_sum(y_pred*HW, axis=1)
    last = y_true**2

    return tf.reduce_mean(var + middle + last)

def entropy_div_bin_met(y_true, y_pred):
    pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])

    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.nn.softmax(y_pred, axis=1)/pdf
    y_pred = tf.transpose(tf.transpose(y_pred) / tf.reduce_sum(y_pred, axis=1))
    y_true = tf.cast(y_true, tf.int32)
    entropy = tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=1)
    entropy = -tf.reduce_mean(entropy)
    return entropy

def KL_binom(y_true, y_pred):
    pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
    y_true = tf.cast(y_true, tf.int32)
    kl = tf.reduce_sum(pdf * tf.math.log(pdf / y_pred), axis=1)
    kl = tf.reduce_mean(kl)
    return kl

def entropy_met(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
    y_true = tf.cast(y_true, tf.int32)
    entropy = tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=1)
    entropy = -tf.reduce_mean(entropy)
    return entropy

def con_entropy_met(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred = tf.nn.softmax(y_pred, axis=1)
    y_true = tf.cast(y_true, tf.int32)
    one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
    target = tf.reduce_sum(y_pred * one_hot, axis=1)

    ratio = tf.clip_by_value(tf.transpose(tf.transpose(y_pred) / (1-target)), 1e-10, 1)
    entropy = tf.reduce_sum(ratio * (1-one_hot) * tf.math.log(ratio), axis=1)
    entropy = -tf.reduce_mean(entropy)
    return entropy

class VarLoss(tf.keras.losses.Loss):
    def __init__(self, name='VarLoss', **kwargs):
        super(VarLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        HW = np.arange(9)
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=1)
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        var = tf.reduce_sum(y_pred * HW * HW, axis=1)
        middle = -2 * y_true * tf.reduce_sum(y_pred*HW, axis=1)
        last = y_true**2

        return tf.reduce_mean(var + middle + last)

class ConEnLoss(tf.keras.losses.Loss):
    def __init__(self, name='ConEnLoss', **kwargs):
        super(ConEnLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        ratio = tf.transpose(tf.transpose(y_pred) / (1-target))
        entropy = tf.reduce_sum(ratio * (1-one_hot) * tf.math.log(ratio), axis=1)
        entropy = -tf.reduce_mean(entropy)
        return -tf.reduce_mean(tf.math.log(target)) + tf.clip_by_value(entropy, 0, 20)

class BinCELoss(tf.keras.losses.Loss):
    def __init__(self, name='CERLoss', **kwargs):
        super(BinCELoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=1)*pdf
        y_pred = tf.transpose(tf.transpose(y_pred) / tf.reduce_sum(y_pred, axis=1))

        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        return -tf.reduce_mean(tf.math.log(target))

class CE_VarLoss(tf.keras.losses.Loss):
    def __init__(self, name='CE_VarLoss', **kwargs):
        super(CE_VarLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        HW = np.arange(9)
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=1)

        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        var = tf.reduce_sum(y_pred * HW * HW, axis=1)
        middle = -2 * y_true * tf.reduce_sum(y_pred*HW, axis=1)
        last = y_true**2

        return -tf.reduce_mean(tf.math.log(target)) + tf.reduce_mean(var + middle + last)

class CE_loss(tf.keras.losses.Loss):
    def __init__(self, name='CE_loss', **kwargs):
        super(CE_loss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        dim = y_pred.shape[1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, dim), (-1, dim))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        return -tf.reduce_mean(tf.math.log(target))

class CE_loss_gaussiann_noise(tf.keras.losses.Loss):
    def __init__(self, name='CE_loss', **kwargs):
        super(CE_loss_gaussiann_noise, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        dim = y_pred.shape[1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, dim), (-1, dim))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)
        #add
        noise = tf.random.normal(shape = tf.shape(target), mean = 0.0, stddev = 0.002)

        return -tf.reduce_mean(tf.math.log(target)) + noise
    
class CE_loss_laplacian_noise(tf.keras.losses.Loss):
    def __init__(self, name='CE_loss', **kwargs):
        super(CE_loss_laplacian_noise, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        dim = y_pred.shape[1]

        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, dim), (-1, dim))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        return -tf.reduce_mean(tf.math.log(target))

class CELoss(tf.keras.losses.Loss):
    def __init__(self, name='CELoss', **kwargs):
        super(CELoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        return -tf.reduce_mean(tf.math.log(target))

class CELoss_comp(tf.keras.losses.Loss):
    def __init__(self, name='CELoss_comp', **kwargs):
        super(CELoss_comp, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.nn.softmax(y_pred, axis=1)/self.pdf
        y_pred = tf.transpose(tf.transpose(y_pred) / tf.reduce_sum(y_pred, axis=1))
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        target = tf.reduce_sum(y_pred * one_hot, axis=1)

        return -tf.reduce_mean(tf.math.log(target))

class CELoss_v2(tf.keras.losses.Loss):
    def __init__(self, name='CELoss_v2', ratio=0, **kwargs):
        super(CELoss_v2, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]

        self.ratio = ratio
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        second = tf.reduce_sum(tf.math.log(y_pred) * (self.pdf - one_hot), axis=1)/(np.sum(self.pdf)-1)
        return tf.reduce_mean(-ce + self.ratio*second)

class CELoss_v2_delta(tf.keras.losses.Loss):
    def __init__(self, name='CELoss_v2_delta', ratio=1, delta=0.1, **kwargs):
        super(CELoss_v2_delta, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]

        self.ratio = ratio
        self.delta = delta
        self.gamma = 0
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        second = tf.reduce_sum(tf.math.log(y_pred) * (self.pdf - one_hot), axis=1)/(np.sum(self.pdf)-1)

        self.gamma += self.delta
        if self.gamma >= 1: self.gamma = 1
        return tf.reduce_mean(-ce + (self.ratio*second)*self.gamma)

class CE_minus_kl_Loss(tf.keras.losses.Loss):
    def __init__(self, name='CE_minus_kl_Loss', ratio=0, **kwargs):
        super(CE_minus_kl_Loss, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]

        self.ratio = 0
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        kl = KL_binom(y_true, y_pred)
        return tf.reduce_mean(-ce) - self.ratio*kl

class CERLoss(tf.keras.losses.Loss):
    def __init__(self, name='CERLoss', **kwargs):
        super(CERLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-10, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        for i in range(100):
            one_hot = tf.reshape(tf.one_hot(tf.random.shuffle(y_true), 9), (-1, 9))
            out = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
            if i == 0: second = out
            else: second += out
        second /= 100

        return tf.reduce_mean(ce) / tf.reduce_mean(second)

class CERLoss_noapprox(tf.keras.losses.Loss):
    def __init__(self, name='CERLoss_noapprox', **kwargs):
        super(CERLoss_noapprox, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(-tf.math.log(y_pred) * one_hot, axis=1)
        E = tf.reduce_sum(-tf.math.log(y_pred) * (self.pdf), axis=1)/(np.sum(self.pdf))

        return tf.reduce_mean(ce) / (tf.reduce_mean(256/255*E - ce/255))

class CERLoss_noapprox_delta(tf.keras.losses.Loss):
    def __init__(self, name='CERLoss_noapprox_delta', delta=0.1, **kwargs):
        super(CERLoss_noapprox_delta, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.gamma = 0
        self.delta = delta
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(-tf.math.log(y_pred) * one_hot, axis=1)
        E = tf.reduce_sum(-tf.math.log(y_pred) * (self.pdf), axis=1)/(np.sum(self.pdf))

        self.gamma += self.delta
        if self.gamma >= 1: self.gamma = 1
        return tf.reduce_mean(ce) / (tf.reduce_mean(256/255*E - ce/255) ** self.gamma)

class CEBLoss(tf.keras.losses.Loss):
    def __init__(self, name='CEBLoss', **kwargs):
        super(CEBLoss, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(-tf.math.log(y_pred) * one_hot, axis=1)
        E = tf.reduce_sum(-tf.math.log(y_pred) * (self.pdf), axis=1)/(np.sum(self.pdf))
        return tf.reduce_mean(ce) / (tf.reduce_mean(E))

class CEBLoss_delta(tf.keras.losses.Loss):
    def __init__(self, name='CEBLoss_delta', delta=0.1, **kwargs):
        super(CEBLoss_delta, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]
        self.gamma = 0
        self.delta = delta
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(-tf.math.log(y_pred) * one_hot, axis=1)
        E = tf.reduce_sum(-tf.math.log(y_pred) * (self.pdf), axis=1)/(np.sum(self.pdf))

        self.gamma += self.delta
        if self.gamma >= 1: self.gamma = 1
        return tf.reduce_mean(ce) / (tf.reduce_mean(E) ** self.gamma)

class CE_KLLoss(tf.keras.losses.Loss):
    def __init__(self, name='CE_KLLoss', **kwargs):
        super(CE_KLLoss, self).__init__(name=name, **kwargs)
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        kl = KL_binom(y_true, y_pred)

        return -tf.reduce_mean(ce) / kl

class CE_CELoss(tf.keras.losses.Loss):
    def __init__(self, name='CE_CELoss', **kwargs):
        super(CE_CELoss, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1), 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)

        kl = tf.reduce_sum(self.pdf * tf.math.log(self.pdf / y_pred), axis=1)
        kl = tf.reduce_mean(kl)

        return -tf.reduce_mean(ce) / (-np.sum(np.log(self.pdf)) + kl)

class CERLoss_comp(tf.keras.losses.Loss):
    def __init__(self, name='CERLoss_comp', **kwargs):
        super(CERLoss_comp, self).__init__(name=name, **kwargs)
        self.pdf = np.array([binom.pmf(i, 8, 1/2) for i in range(9)])
        self.pdf /= self.pdf[0]
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(tf.nn.softmax(y_pred, axis=1)/self.pdf, 1e-20, 1)
        y_true = tf.cast(y_true, tf.int32)
        one_hot = tf.reshape(tf.one_hot(y_true, 9), (-1, 9))
        ce = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
        for i in range(100):
            one_hot = tf.reshape(tf.one_hot(tf.random.shuffle(y_true), 9), (-1, 9))
            out = tf.reduce_sum(tf.math.log(y_pred) * one_hot, axis=1)
            if i == 0: second = out
            else: second += out
        second /= 100

        return tf.reduce_mean(ce) / tf.reduce_mean(second)
