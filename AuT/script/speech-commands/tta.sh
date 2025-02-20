export BASE_PATH=${BASE_PATH:-'/root'}

python -m AuT.speech_commands.tta --dataset 'speech-commands' --dataset_root_path $BASE_PATH'/data/speech_commands' \
    --corrupted_data_root_path $BASE_PATH'/tmp/speech-commands/doing_the_dishes-bg/3.0' \
    --original_auT_weight_path './result/speech-commands/AuT/pre_train/AuT.pt' \
    --original_auC_weight_path './result/speech-commands/AuT/pre_train/AuT-Cls.pt' \
    --max_epoch 20 --lr_cardinality 50 --batch_size 32 --lr '1e-4' --arch 'CT' --arch_level 'base'