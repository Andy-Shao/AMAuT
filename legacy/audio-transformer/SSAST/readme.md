# Software Environment
```shell
conda create --name ssast python=3.9 -y 
conda activate ssast
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.11.3
conda install conda-forge::timm==1.0.12 -y
```

```shell
conda create --name my_test2 python=3.9 -y
conda activate my_test2
# CUDA 11.3
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge
# conda install conda-forge::timm==0.4.5 -y
pip install timm==0.4.5
```

# Cod Reference
+ [SSAST](https://github.com/YuanGongND/ssast/tree/main)