export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.cochlscene.train --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --validation_mode 'validation' --max_epoch 20 --batch_size 32 --arch 'FCE' --arch_level 'base' \
    --lr '1e-3' --background_path $BASE_PATH'/data/speech_commands' --wandb