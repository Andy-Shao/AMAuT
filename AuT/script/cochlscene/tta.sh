#!bin/bash
export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.cochlscene.tta --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --max_epoch 50 --lr_cardinality 50 --batch_size 70 --lr '1e-4' --num_workers 16 --auT_lr_decay 0.55 --auC_lr_decay 1.0 \
    --nucnm_rate 1.0 --lr_gamma 30 --lr_threshold 25 --ent_rate 0.5 --gent_rate 0.5 --gent_q 1.1 --arch 'FCE' --arch_level 'base' \
    --origin_auT_weight './result/CochlScene/AuT/train/FCE-CS-auT.pt' \
    --origin_cls_weight './result/CochlScene/AuT/train/FCE-CS-cls.pt' --wandb