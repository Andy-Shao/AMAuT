export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.vocalsound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --max_epoch 40 --batch_size 32 --lr '5e-4' --arch 'CT' --arch_level 'base' --validation_mode 'validation' \
    --file_name_suffix 'debug' --wandb