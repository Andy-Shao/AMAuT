export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.audiomnist.accu_analysis --dataset 'AudioMNIST2' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --original_auT_weight_path './result/AudioMNIST2/AuT/pre_train/CT-AM2-auT.pt' \
#     --original_auC_weight_path './result/AudioMNIST2/AuT/pre_train/CT-AM2-cls.pt' \
#     --batch_size 32 --arch 'CT' --arch_level 'base'

python -m AuT.audiomnist.accu_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 40 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base'