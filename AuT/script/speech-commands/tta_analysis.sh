export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta_analysis --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'tta_analysis.csv' --batch_size 32 --arch 'CT' --arch_level 'base' \
    --original_auT_weight_path './result/speech-commands/AuT/train/CT-SC-auT.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/train/CT-SC-cls.pt' \
    --original_auT2_weight_path './result/speech-commands/AuT/train/CT-SC-auT2.pt' \
    --original_auC2_weight_path './result/speech-commands/AuT/train/CT-SC-cls2.pt' \
    --original_auT3_weight_path './result/speech-commands/AuT/train/CT-SC-auT3.pt' \
    --original_auC3_weight_path './result/speech-commands/AuT/train/CT-SC-cls3.pt'