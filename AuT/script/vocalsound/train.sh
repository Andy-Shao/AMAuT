export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.vocalsound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_44k' \
    --max_epoch 40 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'test' --wandb