# Software Environment
```shell
conda create --name hdemucs python=3.9 -y
conda activate my-audio
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install conda-forge::transformers==4.46.3
conda install jupyter -y
```