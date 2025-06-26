#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.audiomnist.tta --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --fold 0 --max_epoch 50 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.55 \
    --auC_lr_decay 1.0 --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 25 --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 \
    --arch 'FCE' --arch_level 'base' \
    --origin_auT_weight './result/AudioMNIST/AuT/train/0/FCE-AM-auT0.pt' \
    --origin_cls_weight './result/AudioMNIST/AuT/train/0/FCE-AM-cls0.pt' --wandb