# Software Environment
```shell
conda create --name ssast python=3.9 -y
conda activate ssast
# CUDA 11.3
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y
# conda install conda-forge::timm==0.4.5 -y
pip install timm==0.4.5
```

# Cod Reference
+ [SSAST](https://github.com/YuanGongND/ssast/tree/main)
