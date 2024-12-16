#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[3],sls-1080-3,sls-sm-5
##SBATCH -p gpu
##SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="ssast_pretrain"
#SBATCH --output=./slurm_log/log_%j.txt

set -x
export BASE_PATH=${BASE_PATH:-/home/andyshao}
# comment this line if not running on sls cluster
. $BASE_PATH/data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/sslast2/sslast2/bin/activate
export TORCH_HOME=../../pretrained_models
mkdir exp
mkdir slurm_log

task=pretrain_joint
mask_patch=400