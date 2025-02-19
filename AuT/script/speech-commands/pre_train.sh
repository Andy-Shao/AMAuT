export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.speech_commands.pre_train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --output_csv_name 'training_records.csv' --output_weight_prefix 'speech-commands' --max_epoch 45 --lr_cardinality 50\
#     --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb

python -m AuT.speech_commands.pre_train --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
    --output_csv_name 'training_records.csv' --output_weight_prefix 'speech-commands_v2' --max_epoch 45 --lr_cardinality 50\
    --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb