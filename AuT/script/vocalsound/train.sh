export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.vocalsound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --max_epoch 60 --batch_size 32 --lr '5.5e-4' --arch 'CT' --arch_level 'base' --validation_mode 'validation' \
#     --lr_offset 1 --lr_cardinality 180 --accu_threshold 42.0 --wandb

# python -m AuT.vocalsound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --max_epoch 60 --batch_size 32 --lr '5.5e-4' --arch 'CT' --arch_level 'base' --validation_mode 'validation' \
#     --lr_offset 1 --lr_cardinality 180 --accu_threshold 42.0 --file_name_suffix '2'

# python -m AuT.vocalsound.train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --max_epoch 60 --batch_size 32 --lr '5.5e-4' --arch 'CT' --arch_level 'base' --validation_mode 'validation' \
#     --lr_offset 1 --lr_cardinality 180 --accu_threshold 42.0 --file_name_suffix '3'

python -m AuT.vocalsound.fce_train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --max_epoch 35 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'validation' \
    --lr_cardinality 80 --wandb