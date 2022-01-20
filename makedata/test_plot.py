
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def normalize(data):
    data = data.astype(np.float)
    data -= data.mean()
    print(data.std())
    data /= data.std()
    return data
plt.figure(figsize = (30,8))
#data = np.fromfile('./trace/AES_TI_wave_400000_sakurax_10round', np.uint8)#uint8で
data = np.fromfile('./trace/AES_nonTI_v1_0-500000', np.uint8).astype(np.float)
#data = np.load('./trace/AES_TI_wave_mask_v4_norm_500_900.npy')
wsize = 4800
data = data.reshape(-1,wsize)
train_size = 100
wavel = 0
waver = wsize
data = data[:train_size, wavel:waver]
#a = normalize(data[0])
#a = signal.decimate(data[0],2)
#a =data[0]
print(data.shape)

plt.plot(data[0])
plt.grid()
plt.show()
plt.savefig("./image/test1_6.png")
