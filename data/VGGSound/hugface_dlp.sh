export BASE_PATH=${BASE_PATH:-'/root'}

root_path=$BASE_PATH'/data/VGGSound'
mkdir -p root_path

wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/README.md?download=true -O $root_path'/README.md'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound.csv?download=true -O $root_path'/vggsound.csv'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_00.tar.gz?download=true -O $root_path'/vggsound_00.tar.gz'