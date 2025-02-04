export BASE_PATH=${BASE_PATH:-'/root'}

python -m MsT.speech_commands.MeT_PT --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'training_records.csv' --output_weight_prefix 'speech-commands' --max_epoch 40 \
    --batch_size 32 --lr '1e-3' --arch 'MeT' --embed_size 768