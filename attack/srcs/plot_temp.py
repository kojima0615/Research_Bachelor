import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
kind = "SC"

basedir = "./est_res"
#model ="scnn_9_elu_symmetric_8"
#model= "wouters_AES_HD"
#model = "zaid_ASCAD"
#model ="CNN2_8_plainwave_tanh"
model="CNN5_4_plainwave_tanh_pool2_sub1_mul3"
#model ="CNN3_4_comp"
#dataset = "AES_TI_sakurax_mask_6000000_v5_down3_0005"
#dataset = "AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_hd_lr0001" 
dataset = "AES_TI_wave_mask_div255_v8_0-1500000_1200_1700_memmap_xor_lr0001"
#dataset = "AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_square_xor_lr0001"
#dataset = "AES_TI_wave_mask_norm_vertical_v8_0-1500000_1650_1700_memmap_xor_lr0001"
loss = "CELoss"
#loss = "CE_loss_gaussiann_noise"
epochs = 110
average = 100
num_traces =100000
batch_size = 512
binom = 0

for i in range(1):
    name1 = "b{}.e{}.c{}_{}.npy".format(batch_size, epochs, binom, kind)
    data = np.load(basedir + "/" +dataset + "/" + model + "/" + loss + "/" +name1)
    print(data)
    data = data[:num_traces]
    #plt.plot(data, label="original_CNN_plainwave")
    plt.plot(data, label = "model GE" )
'''
dataset1 = "AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_square_hd_lr0001" 
loss1 = "CELoss"
epochs1 = 48
average1 = 100
num_traces1 = 150000
batch_size1 = 512
binom1 = 0
kind1 = "SC"
model1 ="zaid_ASCAD"
for i in range(1):
    name2 = "b{}.e{}.c{}_{}.npy".format(batch_size1, epochs1, binom1, kind1)
    data = np.load(basedir + "/" +dataset1 + "/" + model1 + "/" + loss1 + "/" +name2)
    print(data)
    #data = data[:]
    plt.plot(data, label="CNN_squarewave")
'''

'''
dataset1 = "AES_nonTI_wave_v3_0_500000_0_1000_xor" 
loss1 = "CELoss"
epochs1 = 11
average1 = 100
num_traces1 =30000
batch_size1 = 256
binom1 = 0
for i in range(1):
    name2 = "b{}.e{}.c{}_{}.npy".format(batch_size1, epochs1, binom1, kind)
    data = np.load(basedir + "/" +dataset1 + "/" + model + "/" + loss1 + "/" +"256256class" + "/" +name2)
    print(data)
    plt.plot(data)
'''
'''
#cpa_sc
ave = 100
base ="/home/usrs/kenta/work/datasets/AES_TI_sakurax" 
cpa_corr = np.load(base + '/csv_data/cpa_0218_mask_TI_time_SC_v8_onevalue.npy')
sc = np.zeros(300000//10000)
for i in range(ave):
    sc += (cpa_corr[i]==0)
sc /= ave
plt.plot(sc, label = "2nd_order_cpa")

plt.plot(data, label="CNN_squarewave")
'''
ymin = 0
if kind == "GE" or kind == "GEtwov":
    ymax = 300
else:
    ymax = 1.1
plt.xlim(xmin=0,xmax=num_traces)
plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig("./image/ge_TI_mask_2021_2_18_wide50_xor_wouters_HD_ep171")
#plt.savefig("./image/sc_TI_2021_2_12_ep166_nonsquare_wide50_CNN5_16_v8")
#plt.savefig("./image/sc_TI_mask_2021_2_19_wide50_xor_norm_CNN3_4_comp_ep613")