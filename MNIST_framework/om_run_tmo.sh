#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0,45,72,75,118,124,128,129,148,400,445,472,475,488,518,524,528,529,548
#SBATCH --gres=gpu:tesla-k80:1
#SBATCH --job-name=avg_r2r
#SBATCH --mem=20GB
#SBATCH -t 08:00:00
#SBATCH --partition=cbmm

module add openmind/singularity/3.4.1

offset_array=(6781)
repetition=("repetition_1" "repetition_2")

for j in "${repetition[@]}"
do
  for i in "${offset_array[@]}"
  do
    singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
      python /om/user/vanessad/MNIST_framework/main.py \
      --host_filesystem om \
      --experiment_index ${SLURM_ARRAY_TASK_ID} \
      --offset_index $i \
      --run train \
      --repetition_folder_path $j
  done
done