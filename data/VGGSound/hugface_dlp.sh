export BASE_PATH=${BASE_PATH:-'/root'}

root_path=$BASE_PATH'/data/VGGSound'
mkdir -p $root_path
cd $root_path

wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/README.md?download=true -O 'README.md'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound.csv?download=true -O 'vggsound.csv'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_00.tar.gz?download=true -O 'vggsound_00.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_01.tar.gz?download=true -O 'vggsound_01.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_02.tar.gz?download=true -O 'vggsound_02.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_03.tar.gz?download=true -O 'vggsound_03.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_04.tar.gz?download=true -O 'vggsound_04.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_05.tar.gz?download=true -O 'vggsound_05.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_06.tar.gz?download=true -O 'vggsound_06.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_07.tar.gz?download=true -O 'vggsound_07.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_08.tar.gz?download=true -O 'vggsound_08.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_09.tar.gz?download=true -O 'vggsound_09.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_10.tar.gz?download=true -O 'vggsound_10.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_11.tar.gz?download=true -O 'vggsound_11.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_12.tar.gz?download=true -O 'vggsound_12.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_13.tar.gz?download=true -O 'vggsound_13.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_14.tar.gz?download=true -O 'vggsound_14.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_15.tar.gz?download=true -O 'vggsound_15.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_16.tar.gz?download=true -O 'vggsound_16.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_17.tar.gz?download=true -O 'vggsound_17.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_18.tar.gz?download=true -O 'vggsound_18.tar.gz'
wget https://huggingface.co/datasets/Loie/VGGSound/resolve/main/vggsound_19.tar.gz?download=true -O 'vggsound_19.tar.gz'

# rclone upload
# rclone copy VGGSound/vggsound_00.tar.gz gdrive:/Datasets/VGGSound/ -P --drive-chunk-size 512M -vv