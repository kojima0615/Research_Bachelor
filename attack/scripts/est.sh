#!/bin/zsh

#model=$1
#model=(scnn_9_elu_symmetric_8)
#model=(wouters_AES_HD)
#model=(CNN2_8_plainwave_tanh)
model=(CNN5_4_plainwave_tanh_pool2_sub1_mul3)
#model=(zaid_ASCAD)

dataset=(AES_TI_wave_mask_div255_v8_0-1500000_1200_1700_memmap_xor_lr0001)
#dataset=(AES_TI_wave_mask_v8_0-1500000_1650_1700_memmap_xor_lr0001)
#dataset=(AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_xor_lr0001)
#dataset=(AES_TI_wave_mask_norm_vertical_v8_0-1500000_1650_1700_memmap_xor_lr0001)
#dataset=(AES_TI_wave_mask_norm_v8_0-1500000_1650_1700_square_xor_lr0001)

loss=(CELoss)
#loss=(CE_loss_gaussiann_noise)
binom=(0)
batch=(512)
epoch=(110)

num_traces=100000
average=100

elem=b$batch.e$epoch.c$binom
target=$dataset/${model}/$loss
mod_file=out/$target/$batch.$epoch.h5
log_dir=est_log/$target
npy_dir=est_res/$target


mkdir -p $log_dir
mkdir -p $npy_dir

#python3 -u ./srcs/calc_preds.py $mod_file \
python3 -u ./srcs/predict_memmap.py $mod_file \
    -d $dataset \
    -o $npy_dir/$elem.npy \
    -b $binom \
    -n $num_traces \
    -a $average > $log_dir/$elem.log