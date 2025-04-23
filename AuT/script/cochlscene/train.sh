export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.cochlscene.train --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --validation_mode 'validation' --max_epoch 30 --batch_size 32 --arch 'CT' --arch_level 'base' \
    --lr '1e-3' --wandb