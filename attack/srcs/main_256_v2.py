import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
import cycler
import numpy as np
import sys
from model import * 
from model_v2 import * 
from model_v3 import * 
from model_comp import *
from tensorflow_model_optimization.sparsity import keras as sparsity
from loss import *
import load_data
import shutil
import os
import matplotlib.pyplot as plt
from waveSequence_v2 import *
from waveSequence_noshuffle_v2 import *
#from keras_radam import RAdam
#export CUDA_VISIBLE_DEVICES=3
def pruning_summary(model):

    for i, w in enumerate(model.get_weights()):
        print(
            "{} -- Total:{}, Zeros: {:.2f}%".format(
                model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
            )
        )


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #使用可能なGPUを列挙
    assert len(physical_devices) > 0, "Not enough GPU"
    #一つもなかったらエラー
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    dataset = sys.argv[1]
    model_name = sys.argv[2]
    Activate_func = sys.argv[3]
    batch_size = int(sys.argv[4])
    tf.random.set_seed(10101)
    epochs = int(sys.argv[5])
    ratio = 1.0

    #model: k.Model = eval(model_name + "({},{})".format(9730, 9))
    model: k.Model = eval(model_name + "({},{})".format(500, 256))
    #{}にtrain_x.shape[1]をいれる model.pyからモデルを定義
    
    #出力する場所を決めていく
    output = dataset + "/" + model_name + "/" + Activate_func
    
    
    if Activate_func == "CELoss_v2" or Activate_func =="CE_minus_kl_Loss":
        output += "_{}".format(ratio)
    output += "/" + str(batch_size)
    
    #前のlogを削除
    shutil.rmtree('logs/' + output, ignore_errors=True)
    
    os.makedirs("out/" + '/'.join(output.split('/')[:-1]), exist_ok = True)
    print(output)
    
    model.compile(
        loss = k.losses.SparseCategoricalCrossentropy(from_logits = True),
        #loss = CE_loss(),
        #loss = CE_loss_gaussiann_noise(),
        optimizer=k.optimizers.Adam(lr = 0.0001),
        metrics =[k.metrics.SparseCategoricalAccuracy('acc')]
        #metrics = [k.metrics.SparseCategoricalAccuracy('acc'), entropy_met, KL_binom, CERLoss_noapprox()]
        )
    print(model.summary())
    #ここにepochごとの出力を入れる
    fpath = "out/" + output + '.{epoch:02d}.h5'
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath = fpath, monitor = 'val_loss', verbose = 1, save_best_only = False, mode = 'auto'),
        tf.keras.callbacks.TensorBoard(log_dir = 'logs/' + output, histogram_freq = 0, write_graph = True),
        tf.keras.callbacks.CSVLogger('logs/' + output +'/log_temp.csv', separator=',', append=False)
        ]
    train_sequence = waveSequence(batch_size,0,1400000)
    test_sequence = waveSequence_sh(batch_size,1400000,1500000)
    #train_sequence = waveSequence(batch_size,0,450000)
    #test_sequence = waveSequence_sh(batch_size,450000,500000)
    history = model.fit_generator(
        generator=train_sequence,
        validation_data = test_sequence,
        epochs=epochs,
        callbacks=callbacks,
        verbose =1,
        workers = 4
        )
    model.save("out/" + output + '.h5')    
    
    
if __name__ == "__main__":
    main()