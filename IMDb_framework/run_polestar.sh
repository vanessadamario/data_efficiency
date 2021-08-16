#!/bin/bash

module add openmind/singularity/3.4.1

for ((i = 0; i <= 143; i++));
  do
  singularity exec -B /om:/om --nv /om/user/xboix/singularity/xboix-tensorflow-vanessa.simg \
  python /om/user/vanessad/IMDb_framework/main.py \
  --host_filesystem om \
  --experiment_index $i \
  --offset_index 445 \
  --run train \
  --repetition_folder_path 1_small_epochs
done