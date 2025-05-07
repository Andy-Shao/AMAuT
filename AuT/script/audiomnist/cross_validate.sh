export BASE_PATH=${BASE_PATH:-'/root'}

# ## fold 4
# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '0' --fold 4

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '1' --fold 4

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '2' --fold 4

# python -m AuT.audiomnist.tta_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'CT' --arch_level 'base' --fold 4 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/4/CT-AM-cls2.pt' \
#     --output_csv_name 'tta_analysis_f4.csv'

# ## fold 1
# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '0' --fold 1

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '1' --fold 1

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '2' --fold 1

# python -m AuT.audiomnist.tta_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'CT' --arch_level 'base' --fold 1 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/1/CT-AM-cls2.pt' \
#     --output_csv_name 'tta_analysis_f1.csv'

# ## fold 2
# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '0' --fold 2

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '1' --fold 2

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '2' --fold 2

# python -m AuT.audiomnist.tta_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'CT' --arch_level 'base' --fold 2 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/2/CT-AM-cls2.pt' \
#     --output_csv_name 'tta_analysis_f2.csv'

# ## fold 3
# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '0' --fold 3

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '1' --fold 3

# python -m AuT.audiomnist.train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'CT' --arch_level 'base' --validation_mode 'validate' \
#     --file_name_suffix '2' --fold 3

# python -m AuT.audiomnist.tta_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'CT' --arch_level 'base' --fold 3 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/3/CT-AM-cls2.pt' \
#     --output_csv_name 'tta_analysis_f3.csv'

############
# FCE
## fold 1
# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 1 --file_name_suffix '0' --wandb

# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 1 --file_name_suffix '1'

# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 1 --file_name_suffix '2'

# python -m AuT.audiomnist.fce_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'FCE' --arch_level 'base' --fold 1 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/1/FCE-AM-cls2.pt' \
#     --output_csv_name 'fce_analysis_f1.csv'
# ## fold 2
# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 2 --file_name_suffix '0' --wandb

# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 2 --file_name_suffix '1'

# python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
#     --fold 2 --file_name_suffix '2'

# python -m AuT.audiomnist.fce_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
#     --batch_size 32 --arch 'FCE' --arch_level 'base' --fold 2 \
#     --original_auT_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-auT0.pt' \
#     --original_auC_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-cls0.pt' \
#     --original_auT2_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-auT1.pt' \
#     --original_auC2_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-cls1.pt' \
#     --original_auT3_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-auT2.pt' \
#     --original_auC3_weight_path './result/AudioMNIST/AuT/train/2/FCE-AM-cls2.pt' \
#     --output_csv_name 'fce_analysis_f2.csv'
## fold 3
python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 3 --file_name_suffix '0' --wandb

python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 3 --file_name_suffix '1'

python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 3 --file_name_suffix '2'

python -m AuT.audiomnist.fce_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --arch 'FCE' --arch_level 'base' --fold 3 \
    --original_auT_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-auT0.pt' \
    --original_auC_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-cls0.pt' \
    --original_auT2_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-auT1.pt' \
    --original_auC2_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-cls1.pt' \
    --original_auT3_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-auT2.pt' \
    --original_auC3_weight_path './result/AudioMNIST/AuT/train/3/FCE-AM-cls2.pt' \
    --output_csv_name 'fce_analysis_f3.csv'
## fold 4
python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 4 --file_name_suffix '0' --wandb

python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 4 --file_name_suffix '1'

python -m AuT.audiomnist.fce_train --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --max_epoch 30 --batch_size 32 --lr '1e-3' --arch 'FCE' --arch_level 'base' --validation_mode 'test' \
    --fold 4 --file_name_suffix '2'

python -m AuT.audiomnist.fce_analysis --dataset 'AudioMNIST' --dataset_root_path $BASE_PATH'/data/AudioMNIST/data' \
    --batch_size 32 --arch 'FCE' --arch_level 'base' --fold 4 \
    --original_auT_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-auT0.pt' \
    --original_auC_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-cls0.pt' \
    --original_auT2_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-auT1.pt' \
    --original_auC2_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-cls1.pt' \
    --original_auT3_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-auT2.pt' \
    --original_auC3_weight_path './result/AudioMNIST/AuT/train/4/FCE-AM-cls2.pt' \
    --output_csv_name 'fce_analysis_f4.csv'