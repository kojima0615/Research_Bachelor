import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
kind = "SC"
'''
basedir = "./logs"
model ="zaid_ASCAD_N100"
dataset = "ASCAD_with_mask"
loss = "CELoss"

batch_size = 50

name = "hamming_weight"
data = pd.read_csv(basedir + "/" +dataset + "/" + model + "/" + loss +"/" + "{}".format(batch_size) + "/log_temp.csv")
print(data.head())
data.plot()
data.plot(x='epoch')
ax=data.plot(y='loss', label ='hamming_weight loss')
data.plot(y='val_loss',ax=ax, label ='hamming_weight val_loss')
'''
basedir = "./logs"
#model ="zaid_ASCAD_N50_key"
#model ="CNN_2"
#model="CNN2_8_plainwave_tanh"
#model="model_base"
model = "CNN5_4_plainwave_tanh_pool2_sub1_mul3"
#model="CNN5_8_plainwave_tanh_pool2"
#model="CNN2_8_plainwave"
#model="wouters_AES_HD_plainwave"
dataset = "AES_TI_wave_mask_div255_v8_0-1500000_1200_1700_memmap_xor_lr0001"
#dataset = "ASCAD_with_mask"
loss = "CELoss"
#loss = "CE_loss_gaussiann_noise"
plt.figure(figsize = (10,8))
batch_size = 512

#name = "hamming_dist"
data1 = pd.read_csv(basedir + "/" +dataset + "/" + model + "/" + loss +"/" + "{}".format(batch_size) + "/log_temp.csv")
ax = data1.plot(y='loss',label ='loss')
data1.plot(y='val_loss',ax=ax,label ='val_loss')
ymin=5.538
ymax=5.552
#ymin = 1.76
#ymax = 1.77
#ymin= 5.5
#ymax= 5.58
plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig("./image/train_loss_AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_memmap_xor_lr0001_CNN3_4_comp")
