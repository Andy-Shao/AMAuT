export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.fce_analysis --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --output_csv_name 'fce_analysis.csv' --batch_size 32 --arch 'FCE' --arch_level 'base' \
    --original_auT_weight_path './result/speech-commands/AuT/train/FCE-SC-auT.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/train/FCE-SC-cls.pt' \
    --original_auT2_weight_path './result/speech-commands/AuT/train/FCE-SC-auT2.pt' \
    --original_auC2_weight_path './result/speech-commands/AuT/train/FCE-SC-cls2.pt' \
    --original_auT3_weight_path './result/speech-commands/AuT/train/FCE-SC-auT3.pt' \
    --original_auC3_weight_path './result/speech-commands/AuT/train/FCE-SC-cls3.pt'

# python -m AuT.speech_commands.fce_analysis --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --output_csv_name 'fce_analysis.csv' --batch_size 32 --arch 'FCE' --arch_level 'base' \
#     --original_auT_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-auT.pt' \
#     --original_auC_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-cls.pt' \
#     --original_auT2_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-auT2.pt' \
#     --original_auC2_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-cls2.pt' \
#     --original_auT3_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-auT3.pt' \
#     --original_auC3_weight_path './result/speech-commands_v2/AuT/train/FCE-SC2-cls3.pt'