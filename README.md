# Audio-Transform

## Project Structure
+ **CoNMix**: the CoNMix test-time training algorithm implement
+ **AuT**: the Audio Transform structure

## Software Environment
```shell
conda create --name my-audio python=3.9 -y 
conda activate my-audio
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -y -c anaconda scipy==1.11.3
conda install conda-forge::ml-collections==0.1.1 -y
conda install pandas==2.2.2 -y
conda install tqdm==4.66.4 -y
# conda install conda-forge::mir_eval==0.6 -y
conda install jupyter -y
conda install matplotlib==3.8.4 -y 
pip install wandb==0.17.1
```
Analysis environment
```shell
conda create --name my-analysis python=3.9 -y
conda activate my-analysis
conda install conda-forge::tensorboard
```

## Processing

## Dataset

### Speech Commands Dataset v0.01
The dataset (1.4 GB) has 65,000 one-second long utterances of 30 short words by thousands of different people, contributed by public members through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in the current implementation
it is marked by the `True` value of `the "is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.

+ Sample size: 64721 (train: 51088, test: 6835, validation: 6798)
+ sample rate: 16000
+ sampel data shape: [1, 5945 - 16000]

|backgroud noise type|sample data shape|sample rate|
|--|--|--|
|doing_the_dishes|[1, 1522930]|16000|
|dude_miaowing|[1, 988891]|16000|
|exercise_bike|[1, 980062]|16000|
|pink_noise|[1, 960000]|16000|
|running_tap|[1, 978488]|16000|
|white_noise|[1, 960000]|16000|

[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)<br/>
[TensorFlow Document](https://www.tensorflow.org/datasets/community_catalog/huggingface/speech_commands)

## Code Reference
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [Audio Clasification in TTA](https://github.com/Andy-Shao/Test-time-Adaptation-in-AC)
