#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=0-99
#SBATCH --job-name=tr_200
#SBATCH --mem=7GB
#SBATCH -t 00:30:00
#SBATCH --partition=cbmm
#SBATCH -D /om/user/vanessad/MNIST_framework/data_gen


hostname
module add openmind/singularity/3.4.1

singularity exec -B /om5:/om5 --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
  python natural_segmented_mnist.py ${SLURM_ARRAY_TASK_ID}
