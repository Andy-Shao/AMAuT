export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.audiomnist.tta_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --arch 'CT' --arch_level 'base' --fold 0 \
    --original_auT_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-auT0.pt' \
    --original_auC_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-cls0.pt' \
    --original_auT2_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-auT1.pt' \
    --original_auC2_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-cls1.pt' \
    --original_auT3_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-auT2.pt' \
    --original_auC3_weight_path './result/AudioMNIST/AuT/train/0/CT-AM-cls2.pt' \
    --output_csv_name 'tta_analysis_f0.csv'