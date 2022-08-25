#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=$(((RANDOM % 1000 + 5000)))
num_gpus=$(awk -F '[0-9]' '{print NF-1}' <<<"$CUDA_VISIBLE_DEVICES")

# Experiments

##################
#   placeholder  #
##################

now=$(date +'%b%d-%H')

experiment_name=$1

if [ -z $experiment_name ]; then
  job_dir=train_${now}
else
  job_dir=train_${experiment_name}_${now}
fi

if [ -d "runs/$job_dir" ]; then
    printf '%s\n' "Removing runs/$job_dir"
    rm -rf "runs/$job_dir"
fi


printf '%s\n' "Training on GPU ${CUDA_VISIBLE_DEVICES}"

srun python -m torch.distributed.run  --nproc_per_node $num_gpus --master_port $MASTER_PORT train.py
##################
#   placeholder  #
##################