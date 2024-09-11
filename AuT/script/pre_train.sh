export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.pre_train --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' --max_epoch 70 \
    --interval 1 --batch_size 64 --trte full --temporary_path $BASE_PATH'/tmp/AudioMNIST_train' \
    --test_rate 0.3 --lr_gamma 10.0 --wandb