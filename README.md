# Augmentation-based Multiview Audio Transformer (AMAuT) Framework for Audio Classification

## Project Structure
+ **AuT**: the Audio Transformer framework
+ **lib**: library
+ **result**: training and analysis results

## Software Environment
Machine image: nvidia/cuda:11.8.0-devel-ubuntu22.04
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
<!--
Analysis environment
```shell
conda create --name my-analysis python=3.9 -y
conda activate my-analysis
conda install conda-forge::tensorboard
```-->

## Processing

### Preparing
```shell
export BASE_PATH=${the parent directory of the project}
conda activate my-audio
cd Audio-Transform
```

### Training
```shell
ssh AuT/script/speech-commands/train.sh
```

### Analysis
```shell
ssh AuT/script/speech-commands/tta_analysis.sh
```

## Dataset

### AudioMNIST
This repository contains code and data used in Interpreting and Explaining Deep Neural Networks for Classifying Audio Signals. The dataset consists of 30,000 audio samples of spoken digits (0–9) from 60 different speakers. Additionally, it holds the audioMNIST_meta.txt, which provides meta information such as the gender or age of each speaker.

+ Sample size: 30000 (Train: 18000, Validation: 6000, Test: 6000)
+ Sample rate: 48000
+ Audio length: 1 second
+ Class Number: 10
<!-- + sample data shape: [1, 14073 - 47998] -->
  
[Audio MNIST Link](https://github.com/soerenab/AudioMNIST/tree/master)

### SpeechCommands v0.01
The dataset (1.4 GB) has 65,000 one-second long utterances of 30 short words by thousands of different people, contributed by public members through the AIY website. This is a set of one-second .wav audio files, each containing a single spoken English word.

In both versions, ten of them are used as commands by convention: "Yes", "No", "Up", "Down", "Left",
"Right", "On", "Off", "Stop", "Go". Other words are considered to be auxiliary (in the current implementation
it is marked by the `True` value of `the "is_unknown"` feature). Their function is to teach a model to distinguish core words
from unrecognized ones.

+ Sample size: 64721 (train: 51088, test: 6835, validation: 6798)
+ Sampling rate: 16000
<!-- + Sample data shape: [1, 5945 - 16000] -->
+ Class Number: 30
+ Audio length: 1 second

[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Dataset Download Link](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

### SpeechCommands v0.02
Add five new words in the Dataset, such as, 'backward', 'forward', 'follow', 'learn', and 'visual'.

+ Sample size: 105829 (train: 84843, test: 11005, validation: 9981)
+ Sampling rate: 16000
+ Class Number: 35
+ Audio length: 1 second
  
[Speech Commands Dataset Link](https://research.google/blog/launching-the-speech-commands-dataset/)<br/>
[Pytorch Document](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html)

### VocalSound
VocalSound is a free dataset consisting of 21,024 crowdsourced recordings of laughter, sighs, coughs, throat clearing, sneezes, and sniffs from 3,365 unique subjects. The VocalSound dataset also contains meta-information such as speaker age, gender, native language, country, and health condition.

+ Sample Size: 20977 (Train: 15531, validation: 1855, test: 3591)
+ Sample rate: 16000
+ Audio length: less than 12 seconds
+ Class number: 6

[VocalSound Dataset Link](https://sls.csail.mit.edu/downloads/vocalsound/)<br/>
Download command:
```shell
wget -O vocalsound_16k.zip https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1
```

### CochlScene
Cochl Acoustic Scene Dataset, or CochlScene, is a new acoustic scene dataset whose recordings are fully collected from crowdsourcing participants. Most of the initial plans and guidelines for the processes were provided by the researchers in the field of audio signal processing and machine learning including the authors, and the actual process was performed by using the crowdsourcing platform developed by SelectStar, a Korean crowdsourcing data company. During the process, the initial plans were reinforced and modified from the discussion about the actual difficulty in the collection process. After extracting the subset of the total collections considering the purpose of the data, we collected 76,115 10 seconds files in 13 different acoustic scenes from 831 participants.

+ Sample rate: 44100
+ Sample size: 76115 (Train: 60855, validation: 7573, test: 7687)
+ Audio length: 10 seconds
+ Class Number: 13

[Github Link](https://github.com/cochlearai/cochlscene)<br/>
[Dataset Link](https://zenodo.org/records/7080122)

## Code Reference
+ [CoNMix](https://github.com/vcl-iisc/CoNMix/tree/master)
+ [TransUNet](https://github.com/Beckschen/TransUNet)
+ [Test-time Adaptation on Audio Classification](https://github.com/Andy-Shao/Test-time-Adaptation-in-AC)
