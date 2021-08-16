#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=single_exp
#SBATCH --mem=8GB
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -D /om/user/vanessad/IMDb_framework/
#SBATCH --partition=use-everything

hostname
module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
python /om/user/vanessad/IMDb_framework/main.py \
--experiment_index 0 \
--offset_index 436 \
--host_filesystem om \
--run train \
--repetition_folder_path example_test \
--save_splits_folder /om/user/vanessad/IMDb_framework/split_train_SST2
