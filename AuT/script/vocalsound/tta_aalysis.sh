export BASE_PATH=${BASE_PATH:-'/root'}

# python -m AuT.vocalsound.tta_analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
#     --batch_size 32 --arch 'CT' --arch_level 'base' --output_csv_name 'tta_analysis.csv' \
#     --original_auT_weight_path './result/VocalSound/AuT/train/CT-VS-auT.pt' \
#     --original_auC_weight_path './result/VocalSound/AuT/train/CT-VS-cls.pt' \
#     --original_auT2_weight_path './result/VocalSound/AuT/train/CT-VS-auT2.pt' \
#     --original_auC2_weight_path './result/VocalSound/AuT/train/CT-VS-cls2.pt' \
#     --original_auT3_weight_path './result/VocalSound/AuT/train/CT-VS-auT3.pt' \
#     --original_auC3_weight_path './result/VocalSound/AuT/train/CT-VS-cls3.pt'

python -m AuT.vocalsound.fce_analysis --dataset 'VocalSound' --dataset_root_path $BASE_PATH'/data/vocalsound_16k' \
    --batch_size 32 --arch 'FCE' --arch_level 'base' --output_csv_name 'fce_analysis.csv' \
    --original_auT_weight_path './result/VocalSound/AuT/train/FCE-VS-auT.pt' \
    --original_auC_weight_path './result/VocalSound/AuT/train/FCE-VS-cls.pt' \
    --original_auT2_weight_path './result/VocalSound/AuT/train/FCE-VS-auT2.pt' \
    --original_auC2_weight_path './result/VocalSound/AuT/train/FCE-VS-cls2.pt' \
    --original_auT3_weight_path './result/VocalSound/AuT/train/FCE-VS-auT3.pt' \
    --original_auC3_weight_path './result/VocalSound/AuT/train/FCE-VS-cls3.pt'