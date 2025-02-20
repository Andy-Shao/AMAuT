export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.corrupt_dataset --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_path $BASE_PATH'/tmp/speech-commands/doing_the_dishes-bg/3.0' --data_type 'raw' \
    --rand_bg --corruption 'doing_the_dishes' --severity_level 3.0 --cal_strong