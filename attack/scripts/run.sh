#!/bin/zsh
#datasets=(AES_TI_wave_mask_norm_vertical_v8_0-1500000_1650_1700_memmap_xor_lr0001)
#datasets=(AES_TI_wave_mask_norm_v8_0-1500000_600_1800_memmap_xor_lr0001)
datasets=(AES_TI_wave_mask_div255_v8_0-1500000_1200_1700_memmap_xor_lr0001)
#datasets=(AES_TI_wave_mask_norm_v8_0-1500000_1300_1700_memmap_xor_lr0001)
#datasets=(AES_TI_wave_unmask_norm_v6_0-500000_1650_1700_hd_lr0001)
batch_sizes=(512)
#models=(scnn_9_elu_symmetric_16)
#models=(CNN2_8_plainwave_tanh_mybias)
#models=(CNN2_8_plainwave_tanh_8_8)
#models=(wouters_AES_HD_plainwave)
#models=(CNN5_8_plainwave_tanh_pool2)
models=(model_base_mul1)
act=CELoss
#act=CE_loss_gaussiann_noise

for dataset in $datasets; do
    for model in $models; do
        for batch in $batch_sizes; do
            epoch=500
            if [ "$dataset" = "ASCAD_with_mask" ]; then
                epoch=100
            fi
            #python3 ./srcs/main.py $dataset $model $act $batch $epoch
            nohup python3 ./srcs/main_256_v2.py $dataset $model $act $batch $epoch &
        done
    done
done
