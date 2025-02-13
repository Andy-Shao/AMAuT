export BASE_PATH=${BASE_PATH:-'/root'}

python -m data.VGGSound.download --output_path $BASE_PATH'/data/VGGSound' \
    --meta_file_path $BASE_PATH'/vggsound.csv'