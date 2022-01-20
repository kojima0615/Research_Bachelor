

datasets = (AES_TI_sakurax)

batch_sizes = (50)
model = (zaid_ASCAD_N100_key)
loss = CELoss

#binom 0 正規分布の補正なし　1 あり
for dataset in $datasets; do
    epochs = 50
    if [ "$dataset" = "ASCAD_with_mask" ]; then
        epochs = 100
    fi

    for model in $models; do
        for batch in $batch_sizes; do
            for epoch in {1..$epochs}; do
                epoch = `printf %02d $epoch`
                for binom in {0..1}; do
                    echo $model $dataset $loss $binom $batch $epoch
                    ./scripts/est.sh $model $dataset $loss $binom $batch $epoch
                done
            done
        done
    done
done 