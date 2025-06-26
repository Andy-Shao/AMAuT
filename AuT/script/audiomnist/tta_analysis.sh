export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.audiomnist.fce_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --arch 'FCE' --arch_level 'base' --fold 0 \
    --original_auT_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-auT0.pt' \
    --original_auC_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-cls0.pt' \
    --original_auT2_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-auT1.pt' \
    --original_auC2_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-cls1.pt' \
    --original_auT3_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-auT2.pt' \
    --original_auC3_weight_path './result/AudioMNIST/AuT/train/0/FCE-AM-cls2.pt' \
    --output_csv_name 'fce_analysis_f0.csv'