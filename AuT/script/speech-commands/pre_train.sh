export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT2.speech-commands.pre_train --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'training_records.csv' --output_weight_prefix 'speech-commands' --normalized \
    --max_epoch 20 --interval 20 --batch_size 64 --lr '1e-2'