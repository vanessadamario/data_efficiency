#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --job-name=exp_text
#SBATCH --array=504,505,528,529,559,603
#SBATCH --mem=8GB
#SBATCH
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -D /om/user/vanessad/IMDb_framework/
#SBATCH --partition=use-everything

hostname
module add openmind/singularity/3.4.1

singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
python /om/user/vanessad/IMDb_framework/runs/tmp_load_word2vec.py