export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.vocalsound.fce_train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --max_epoch 35 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'validation' \
#     --lr_cardinality 80 --background_path $BASE_PATH'/data/speech_commands' --wandb

python -m AuT.vocalsound.fce_train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --max_epoch 35 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'validation' \
    --lr_cardinality 80 --background_path $BASE_PATH'/data/speech_commands' --file_name_suffix '2'

python -m AuT.vocalsound.fce_train --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --max_epoch 35 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'validation' \
    --lr_cardinality 80 --background_path $BASE_PATH'/data/speech_commands' --file_name_suffix '3'