import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
kind = "SC"

basedir = "./logs"

model="scnn_9_elu_symmetric_8_twov"
dataset = "AES_TI_sakurax_mask_1000000_v1"
#dataset = "ASCAD_with_mask"
loss = "CELoss"
plt.figure(figsize = (10,8))
batch_size = 256


data1 = pd.read_csv(basedir + "/" +dataset + "/" + model + "/" + loss +"/" + "{}".format(batch_size) + "/log_temp.csv")
ax = data1.plot(y='output1_loss',label ='output1_loss')
data1.plot(y='output2_loss',ax=ax,label ='output2_loss')
data1.plot(y='val_output1_loss',ax=ax,label ='val_output1_loss')
data1.plot(y='val_output2_loss',ax=ax,label ='val_output2_loss')
ymin=5.540
ymax=5.56
plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("./image/train_loss_scnn9_mask_0111_twov.png")