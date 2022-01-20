from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
# matplotlib.use('Agg')

#フォントを変えたいときにはコメントアウトを切ってね
#del font_manager.weight_dict['roman']
#font_manager._rebuild()
#font = font_manager.FontProperties()
#font.set_family('serif')
#font.set_name('Times New Roman')
#font.set_weight('light')
#plt.rcParams["font.family"] = "Times New Roman"      #全体のフォントを設定
plt.figure(figsize = (20,8))
plt.rcParams["xtick.direction"] = "in"               #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams["ytick.direction"] = "in"               #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams["xtick.minor.visible"] = True          #x軸補助目盛りの追加
plt.rcParams["ytick.minor.visible"] = True          #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.5              #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.5              #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 1.0              #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 1.0              #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 10                #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 10                #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 5                #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 5                #y軸補助目盛り線の長さ
plt.rcParams["font.size"] = 22                 #フォントの大きさ
plt.rcParams["axes.linewidth"] = 1.5                 #囲みの太さ
import numpy as np
kind = "GE"

basedir = "./est_res"
model ="scnn_9_elu_symmetric_key"
dataset = "AES_TI_sakurax"
loss = "CELoss"
epochs = 10
average = 10
num_traces = 400000
batch_size = 512
binom = 0
for i in range(1):
    name1 = "b{}.e{}.c{}_{}.npy".format(batch_size, epochs, binom, kind)
    data = np.load(basedir + "/" +dataset + "/" + model + "/" + loss + "/" +name1)
    print(data)
    plt.plot(data)
'''
model ="zaid_ASCAD_N100"
name = "hamming weight without compensation"
name2 = "b{}.e{}.c{}_{}.npy".format(batch_size, epochs, binom, kind)
data = np.load(basedir + "/" +dataset + "/" + model + "/" + loss + "/" +name2)
plt.plot(data, label=name)

binom = 1
epochs = 50
name = "hamming weight with compensation"
name2 = "b{}.e{}.c{}_{}.npy".format(batch_size, epochs, binom, kind)
data = np.load(basedir + "/" +dataset + "/" + model + "/" + loss + "/" +name2)
plt.plot(data, label=name)
'''
ymin = 0
if kind == "GE":
    ymax = 300
else:
    ymax = 1.1

plt.xlim(xmin=0,xmax=num_traces)
plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
#plt.legend()
plt.show()
plt.savefig("./image/ge_sakurax_all_mask_koseki_test")