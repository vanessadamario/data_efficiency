#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --array=17,18,19,20,54,55,56,57,91,92,93,94,128,129,130,131,165,166,167,168,202,203,204,205,239,240,241,242,276,277,278,279,313,314,315,316,350,351,352,353,387,388,389,390
#SBATCH --job-name=save_n100
#SBATCH --gres=gpu:titan-x:1
#SBATCH --mem=12GB
#SBATCH -t 02:30:00
#SBATCH --partition=cbmm
#SBATCH -D /om/user/vanessad/MNIST_framework/slurm_output/output_results_natural/scenario3/repetition_5

hostname
module add openmind/singularity/3.4.1

offset_array=(0)
repetition=("plot_activations")

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