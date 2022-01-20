import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize = (15,10))
kind = "SCtwov"

basedir = "./est_res"
#model ="scnn_9_elu_symmetric_8"
#model= "wouters_ASCAD"
model ="CNN2_8_plainwave_tanh"
#model ="CNN3_4_plainwave"
#dataset = "AES_TI_sakurax_mask_6000000_v5_down3_0005"
#dataset = "AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_hd_lr0001" 
dataset = "AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_xor_lr0001"
loss = "CELoss"
#loss = "CE_loss_gaussiann_noise"
epochs = 250
average = 100
num_traces =120000
batch_size = 512
binom = 0

for i in range(1):
    name1 = "b{}.e{}.c{}_{}.npy".format(batch_size, epochs, binom, kind)
    data = np.load(basedir + "/" +dataset + "/" + model + "/" + loss + "/" +name1)
    print(data)
    data = data[:num_traces]
    plt.plot(data, label="Original CNN Raw waveform")
dataset1 = "AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_square_xor_lr0001" 
loss1 = "CELoss"
epochs1 = 8
average1 = 100
num_traces1 = 200000
batch_size1 = 512
binom1 = 0
kind1 = "SCtwov"
model1 ="zaid_ASCAD"
for i in range(1):
    name2 = "b{}.e0{}.c{}_{}.npy".format(batch_size1, epochs1, binom1, kind1)
    data = np.load(basedir + "/" +dataset1 + "/" + model1 + "/" + loss1 + "/" +name2)
    print(data)
    #data = data[:]
    plt.plot(data, label="CNN Squared waveform")


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

#cpa_sc
ave = 100
base ="/home/usrs/kenta/work/datasets/AES_TI_sakurax" 
cpa_corr = np.load(base + '/csv_data/cpa_0215_mask_TI_time_SC_v8_twovalue.npy')
sc = np.zeros(300000//10000)
print(cpa_corr.shape)
for i in range(ave):
    sc += (cpa_corr[i]==0)
sc /= ave
sc = sc[:20]
#plt.plot(np.linspace(100,150100,1500),sc, label = "2nd_order_cpa")
plt.plot(np.linspace(10000,210000,20),sc, label = "2nd Order CPA")

ymin = 0
if kind == "GE" or kind == "GEtwov":
    ymax = 300
else:
    ymax = 1.1
plt.xlim(xmin=0,xmax=num_traces1)
plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
plt.xlabel("number of waveforms", fontsize=20)
plt.ylabel("success rate", fontsize=20)
plt.tick_params(labelsize=15)
plt.legend(fontsize=20)
#plt.show()
#plt.savefig("./image/sc_TI_unmask_1byte_test_2021_1_30")
#plt.savefig("./image/sc_TI_2021_2_12_ep166_nonsquare_wide50_CNN5_16_v8")
#plt.savefig("./image/sc_TI_2021_2_16_ep145_plainwave_wide50_v8_CNN2_8_plainwave_hd_nonbinom")
plt.savefig("./image/sctwov_TI_mask_2021_2_20_wide50_xor_comp_v8_short")