#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.45 --auC_lr_decay 0.45 \
    --nucnm_rate 0.2 --lr_gamma 30 --lr_threshold 25 --ent_rate 1.0 --gent_rate 0.0 --gent_q 0.8 --arch 'FCE' --arch_level 'base' \
    --origin_auT_weight './result/speech-commands/AuT/train/FCE-SC-auT.pt' \
    --origin_cls_weight './result/speech-commands/AuT/train/FCE-SC-cls.pt' --wandb


# python -m AuT.speech_commands.tta --dataset 'speech-commands_v2' --dataset_root_path $BASE_PATH'/data' \
#     --max_epoch 80 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.45 --auC_lr_decay 0.45 \
#     --nucnm_rate 0.0 --lr_gamma 30 --lr_threshold 25 --ent_rate 1.0 --gent_rate 0.2 --gent_q 1.1 --arch 'FCE' --arch_level 'base' \
#     --origin_auT_weight './result/speech-commands_v2/AuT/train/FCE-SC2-auT.pt' \
#     --origin_cls_weight './result/speech-commands_v2/AuT/train/FCE-SC2-cls.pt' --wandb