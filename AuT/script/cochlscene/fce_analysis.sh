export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.cochlscene.fce_analysis --dataset 'CochlScene' --dataset_root_path $BASE_PATH'/data/CochlScene' \
    --output_csv_name 'fce_analysis' --batch_size 32 --arch 'FCE' --arch_level 'base' \
    --original_auT_weight_path './result/CochlScene/AuT/train/FCE-CS-auT.pt' \
    --original_auC_weight_path './result/CochlScene/AuT/train/FCE-CS-cls.pt' \
    --original_auT2_weight_path './result/CochlScene/AuT/train/FCE-CS-auT2.pt' \
    --original_auC2_weight_path './result/CochlScene/AuT/train/FCE-CS-cls2.pt' \
    --original_auT3_weight_path './result/CochlScene/AuT/train/FCE-CS-auT3.pt' \
    --original_auC3_weight_path './result/CochlScene/AuT/train/FCE-CS-cls3.pt' --only_origin