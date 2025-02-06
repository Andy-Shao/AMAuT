export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta_analysis --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'tta_analysis.csv' --batch_size 32 --arch 'CT' --embed_size 768 \
    --original_auT_weight_path './result/speech-commands/AuT/pre_train/AuT.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/pre_train/AuT-Cls.pt' \
    --data_type 'raw'