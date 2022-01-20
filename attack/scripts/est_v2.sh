#!/bin/zsh

#model=$1
#model=(scnn_9_elu_symmetric_16)
#model=(CNN5_16)
model=(CNN3_4_plainwave)
#model=(zaid_ASCAD)
#dataset=(AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_square_hd_lr0001)
#dataset=(AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_hd_lr0001)
dataset=(AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_xor_lr0001)
#dataset=(AES_TI_wave_mask_v8_0-1500000_1650_1700_square_xor_lr0001)
#loss=(CE_loss_gaussiann_noise)
loss=(CELoss)
binom=(0)
batch=(512)
epoch=(379)

num_traces=120000
average=100

elem=b$batch.e$epoch.c$binom
target=$dataset/${model}/$loss
log_dir=est_log/$target
npy_dir=est_res/$target
pred_dir=$npy_dir/$elem


mkdir -p $log_dir
mkdir -p $npy_dir



nohup python3 -u ./srcs/calc_ge_sc.py $pred_dir.npy \
    -d $dataset \
    -o $npy_dir/$elem.npy \
    -b $binom \
    -n $num_traces \
    -a $average > $log_dir/$elem.log &