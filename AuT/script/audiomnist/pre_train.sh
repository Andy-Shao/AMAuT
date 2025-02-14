export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.audiomnist.pre_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --output_csv_name 'training_records.csv' --output_weight_prefix 'AudioMNIST' --max_epoch 20 \
    --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --wandb