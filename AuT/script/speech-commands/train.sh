export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.speech_commands.train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --max_epoch 45 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb

python -m AuT.speech_commands.train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
    --max_epoch 80 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb