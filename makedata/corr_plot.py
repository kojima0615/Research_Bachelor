import numpy as np
import matplotlib.pyplot as plt
'''
base = "/home/usrs/kenta/work/datasets"
wave = np.fromfile(base + '/AES_TI_sakurax/trace/AES_TI_wave_mask_v3_test', np.uint8)
wave = wave.reshape(-1,4800)
data = wave[0].astype(np.float)
data -= data.mean()
data /=100
'''
corr = np.load("./csv_data/corr_0210_all_square_v8.npy")
print(corr.shape)
corr = corr[0:2000]
corr = corr.T
#print(np.argmax(abs(corr[89])))#849
plt.figure(figsize = (10,10))
for i in range(256):
    if i != 0:
        plt.plot(abs(corr[i]),color='grey')
plt.plot(abs(corr[0]),color='red')
#plt.xlim(xmin=0,xmax=corr.shape[1])
#plt.ylim(ymin=0,ymax=0.2)
plt.grid(True)
plt.legend()
plt.show()
plt.savefig("./image/corr_0221_TI_0_2000_square.png")
#0byte 89
