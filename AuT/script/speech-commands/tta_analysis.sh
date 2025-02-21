export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta_analysis --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --corrupted_data_root_path $BASE_PATH'/tmp/speech-commands/doing_the_dishes-bg/3.0' \
    --output_csv_name 'tta_analysis.csv' --batch_size 32 --arch 'CT' --arch_level 'base' \
    --original_auT_weight_path './result/speech-commands/AuT/pre_train/CT-SC-auT.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/pre_train/CT-SC-cls.pt' \
    --data_type 'raw' --corruption 'doing_the_dishes' --severity_level 3.0