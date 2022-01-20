import numpy as np
from scipy import signal
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
def normalize(data):
    data = data.astype(np.float)
    #data1 = data1.astype(np.float)
    data -= data.mean()
    #print(data.std())
    data /= data.std()
    return data

def div255(data):
    data = data.astype(np.float)
    data/=255
    return data

def normalize_mat(data):
    return np.array([normalize(v) for v in data])

def abdiff(data):
    data = data.astype(np.float)
    res = np.array([])
    for i in range(data.shape[0]-1):
        res = np.concatenate([res, np.abs(data[i+1:data.shape[0]] - data[i])])
    return res

def product_prev(data):
    data = data.astype(np.float)
    
    data = data.T
    for i in range(data.shape[0]):
        data[i]-=data[i].mean()
    data = data.T
    
    return data

def product(data):
    res = np.array([])
    for i in range(data.shape[0]):
        res = np.concatenate([res, data[i:data.shape[0]] * data[i]])
    return res


def square_w(data):
    data = data.astype(np.float)

    #op
    #data -= data.mean()

    data = np.square(data)
    return data


wave_size = 4690 #round1v7
#wave_size = 4690 #round1 v8
#wave_size = 4800 #round10
#wave_size = 4750 #round1 nonTI
#train_traces = np.fromfile('./trace/AES_TI_wave_unmask_v6_test_0-500000', np.uint8)
train_traces = np.fromfile('./trace/AES_TI_wave_mask_v8_test_0-500000', np.uint8)
#train_traces = train_traces[:(train_traces.shape[0]//wave_size) * wave_size]
#train_traces = train_traces.reshape(-1, wave_size)
train_traces = train_traces.reshape(500000, -1)


'''
train_traces_2 = np.fromfile('./trace/AES_TI_wave_mask_v8_500000-1500000', np.uint8)
#train_traces_2 = train_traces_2[:(train_traces_2.shape[0]//wave_size) * wave_size]
train_traces_2 = train_traces_2.reshape(-1, wave_size)
print(train_traces_2.shape)
train_traces = np.concatenate([train_traces, train_traces_2])
del train_traces_2
'''
'''
train_traces0 = np.fromfile('./trace/AES_TI_wave_mask_v8_test_0-500000', np.uint8)
#train_traces = train_traces[:(train_traces.shape[0]//wave_size) * wave_size]
#train_traces = train_traces.reshape(-1, wave_size)
train_traces0 = train_traces0.reshape(500000, -1)
'''
'''
train_traces_2 = np.fromfile('./trace/AES_TI_wave_mask_v8_1500000-2500000', np.uint8)
#train_traces_2 = train_traces_2[:(train_traces_2.shape[0]//wave_size) * wave_size]
train_traces_2 = train_traces_2.reshape(-1, wave_size)
print(train_traces_2.shape)
train_traces = np.concatenate([train_traces, train_traces_2])
del train_traces_2
'''



#predict
'''
train_traces = np.fromfile('./trace/AES_nonTI_v1_test_0-500000', np.uint8)
train_traces = train_traces.reshape(-1, wave_size)
'''

print(train_traces.shape)
#train_traces = train_traces[:1340000,880:1420]
#train_traces = train_traces[:1500000,1650:1700]
train_traces = train_traces[:250000,1200:1700]
print(train_traces.shape)
#train_traces0 = train_traces0[:500000,1650:1700]
#train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(signal.decimate)(v,3) for v in train_traces]))


#product
train_traces = product_prev(train_traces)
train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(product)(v) for v in train_traces]))

'''
#square
train_traces = product_prev(train_traces)
train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(square_w)(v) for v in train_traces]))
'''
#absolute_difference
#train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(abdiff)(v) for v in train_traces]))

'''
#np.save("./trace/AES_TI_wave_mask_v8_test_0-500000_1650_1700",train_traces)
plt.figure(figsize = (30,8))
plt.plot(train_traces[0])
plt.show()
'''
#train_traces = train_traces.T
#train_traces0 = train_traces0.T
#train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(normalize)(v,v1) for v,v1 in zip(train_traces,train_traces0)]))
#train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(normalize)(v) for v in train_traces]))
#train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(div255)(v) for v in train_traces]))
#train_traces0 = train_traces0.T
#plt.plot(train_traces[0])
#plt.show()
#print(train_traces.shape)
print(train_traces.shape)
X_MEMMAP_PATH = './memmap/AES_TI_wave_mask_product_v8_test_0-250000_1200_1700_memmap.npy'

X_memmap = np.memmap(
    filename=X_MEMMAP_PATH, dtype=np.float, mode='w+', shape=(train_traces.shape[0], train_traces.shape[1]))
X_memmap[:] = train_traces
#np.save("./trace/AES_TI_wave_mask_product_v8_test_0-500000_1200_1700",train_traces)

