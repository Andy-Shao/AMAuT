export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'test' \
#     --file_name_suffix 'debug' --wandb

python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
    --file_name_suffix '0'

python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
    --file_name_suffix '1'

python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
    --file_name_suffix '2'