export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.speech_commands.train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --max_epoch 45 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb

# python -m AuT.speech_commands.train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --max_epoch 45 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' \
#     --file_name_suffix '2'

# python -m AuT.speech_commands.train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --max_epoch 45 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' \
#     --file_name_suffix '3'

# python -m AuT.speech_commands.train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --max_epoch 80 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb

# python -m AuT.speech_commands.train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --max_epoch 80 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' \
#     --file_name_suffix '2'

# python -m AuT.speech_commands.train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --max_epoch 80 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' \
#     --file_name_suffix '3'

python -m AuT.speech_commands.fce_train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' \
    --background_path $BASE_PATH'/data/speech_commands' --num_workers 16 --wandb

# python -m AuT.speech_commands.fce_train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --max_epoch 30 --lr_cardinality 50 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' \
#     --background_path $BASE_PATH'/data/speech-commands_v2/speech_commands_v0.02' --num_workers 16 \
#     --wandb