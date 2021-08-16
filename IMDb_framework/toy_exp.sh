#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0-293
#SBATCH --job-name=exp_0
#SBATCH --mem=8GB
#SBATCH -t 00:40:00
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH -D /om/user/vanessad/IMDb_framework/slurm_output/tesla_exp_0
#SBATCH --partition=use-everything

hostname
module add openmind/singularity/3.4.1

folderindex=(0)

for i in "${folderindex[@]}"
do
  singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
  python /om/user/vanessad/IMDb_framework/main.py \
  --experiment_index ${SLURM_ARRAY_TASK_ID} \
  --offset_index 0 \
  --host_filesystem om \
  --run train \
  --repetition_folder_path $i \
  --save_splits_folder /om/user/vanessad/IMDb_framework/split_train_SST2
done