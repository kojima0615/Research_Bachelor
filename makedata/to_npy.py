import numpy as np
from scipy import signal
from joblib import Parallel, delayed

def normalize(data):
    data = data.astype(np.float)
    data -= data.mean()
    print(data.std())
    data /= data.std()
    return data

def normalize_mat(data):
    return np.array([normalize(v) for v in data])


wave_size = 4800
train_traces = np.fromfile('./trace/AES_TI_wave_mask_v4_test', np.uint8)
train_traces = train_traces.reshape(-1, wave_size)
train_traces = np.array(Parallel(n_jobs = 32, verbose=3)([delayed(signal.decimate)(v,2) for v in train_traces]))
print(train_traces.shape)
np.save("./trace/AES_TI_wave_mask_v4_test_down2",train_traces)