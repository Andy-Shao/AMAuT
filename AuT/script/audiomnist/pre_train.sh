export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.audiomnist.pre_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 40 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb

python -m AuT.audiomnist.pre_train --dataset 'AudioMNIST2' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 40 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb