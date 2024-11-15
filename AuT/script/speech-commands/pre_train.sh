export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech-commands.pre_train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'training_records.csv' --output_weight_prefix 'speech-commands' --max_epoch 40 --interval_num 40 \
    --batch_size 64 --lr '1e-2' --wandb --early_stop 20