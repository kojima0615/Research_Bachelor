import tensorflow as tf
k = tf.keras
kl = tf.keras.layers
import cycler
import numpy as np
import sys
from model import * 
from tensorflow_model_optimization.sparsity import keras as sparsity
from loss import *
import load_data
import shutil
import os
import matplotlib.pyplot as plt

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
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    dataset = sys.argv[1]
    model_name = sys.argv[2]
    Activate_func = sys.argv[3]
    batch_size = int(sys.argv[4])
    tf.random.set_seed(10101)
    epochs = int(sys.argv[5])
    ratio = 1.0
    validaton_split = 0.1
    
    (train_x, train_y), (test_x,train_y1) = load_data.traces(dataset,model_name)
    train_y1 = train_y1.astype(np.float)

    model: k.Model = eval(model_name + "({}, {})".format(train_x.shape[1], 256))
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
        #loss = {'output1':k.losses.SparseCategoricalCrossentropy(from_logits = True)
        #,'output2':k.losses.SparseCategoricalCrossentropy(from_logits = True)},
        #loss = CE_loss(),
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
    history = model.fit(
        #x = train_x,
        x = [train_x,train_y1],
        #x = {'input_w':train_x,'input_B':train_y1},
        #y = {'output1':np.array(train_y),'output2':np.array(train_y1)},
        y = train_y,
        batch_size = batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split = validaton_split
        )
    model.save("out/" + output + '.h5')
    
    
if __name__ == "__main__":
    main()