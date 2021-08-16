#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
# SBATCH --job-name=activations
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=20GB
#SBATCH -t 00:20:00
#SBATCH --partition=normal


hostname
module add openmind/singularity/3.4.1


singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow2.simg \
  python /om/user/vanessad/MNIST_framework/runs/activation_mnist_natural.py
