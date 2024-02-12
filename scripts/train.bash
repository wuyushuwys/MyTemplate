#!/bin/bash

export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(nvidia-smi --list-gpus | wc -l)

# Experiments

##################
#   placeholder  #
##################

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=runs/train_${now}
else
  job_dir=runs/train_${experiment_name}_${now}
fi

if [ -d "runs/$job_dir" ]; then
    printf '%s\n' "Removing $job_dir"
    rm -rf $job_dir
fi


printf '%s\n' "Training on GPU ${CUDA_VISIBLE_DEVICES}"

python -m torch.distributed.run  --nproc_per_node $num_gpus --master_port $MASTER_PORT train.py
##################
#   placeholder  #
##################