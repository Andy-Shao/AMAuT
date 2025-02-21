export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --original_auT_weight_path './result/speech-commands/AuT/pre_train/CTA-SC.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/pre_train/CTA-SC-Cls.pt' \
    --original_auD_weight_path './result/speech-commands/AuT/pre_train/CTA-SC-Dec.pt' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --arch 'CTA' --arch_level 'base'

# python -m AuT.speech_commands.tta --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
#     --original_auT_weight_path './result/speech-commands/AuT/pre_train/CT-SC.pt' \
#     --original_auC_weight_path './result/speech-commands/AuT/pre_train/CT-SC-Cls.pt' \
#     --max_epoch 20 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --arch 'CT' --arch_level 'base'