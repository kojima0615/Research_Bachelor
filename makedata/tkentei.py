import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
#train_traces = np.fromfile('./trace/AES_TI_wave_1000000_sakurax_10round', np.uint8).astype(np.float)
trace_data = np.fromfile('./trace/AES_TI_wave_100000_tkentei_unmask', np.uint8).astype(np.float)
w_size = 8910
trace_data = trace_data.reshape(-1,w_size)
train_traces = trace_data.reshape(-1,w_size)
print(trace_data.shape) 

random_trace = trace_data[:50000,:]
#random_trace = train_traces
fixed_trace = trace_data[50000:100000,:]
random_trace = random_trace.T
fixed_trace = fixed_trace.T
print(random_trace.shape)
print(fixed_trace.shape) 

random_mean = [0] * w_size
fixed_mean = [0] * w_size
random_std = [0] * w_size
fixed_std = [0] * w_size
t_value = [0]*w_size
for i in range(w_size):
    '''
    random_mean[i] = random_trace[i].mean()
    random_std[i] = random_trace[i].std()
    fixed_mean[i] = fixed_trace[i].mean()
    fixed_std[i] = fixed_trace[i].std()
    t_value[i] = (fixed_mean[i] - random_mean[i]) / fixed_std[i]
    t_value[i] *= pow(50000,0.5)
    '''
    t_value[i],_ = ttest_ind(random_trace[i], fixed_trace[i])
plt.plot(t_value)
plt.ylim(ymin=-200,ymax=200)
plt.grid(True)
plt.legend()
plt.savefig("./image/tkentei_unmask_powerpoint")

