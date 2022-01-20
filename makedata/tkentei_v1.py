import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#random_trace = np.fromfile('./trace/AES_TI_wave_mask_v3_0-1000000', np.uint8).astype(np.float)
#fixed_trace = np.fromfile('./trace/AES_TI_wave_mask_v3_test', np.uint8).astype(np.float)
random_trace = np.load('./trace/AES_TI_wave_mask_v4_norm.npy')
fixed_trace = np.load('./trace/AES_TI_wave_mask_v4_test_norm.npy')
w_size = 4800
random_trace = random_trace.reshape(-1,w_size)
fixed_trace = fixed_trace.reshape(-1,w_size)
random_trace = random_trace[:10000,:]
fixed_trace = fixed_trace[:10000,:]
random_trace = random_trace.T
fixed_trace = fixed_trace.T
print(random_trace.shape)
print(fixed_trace.shape) 
plt.figure(figsize = (30,8))
random_mean = [0] * w_size
fixed_mean = [0] * w_size
random_std = [0] * w_size
fixed_std = [0] * w_size
t_value = [0]*w_size
for i in range(w_size):
    t_value[i],_ = ttest_ind(random_trace[i], fixed_trace[i])
plt.plot(t_value)
plt.ylim(ymin=-50,ymax=50)
plt.grid(True)
plt.legend()
plt.savefig("./image/tkentei_mask_v4_0123_10000")

