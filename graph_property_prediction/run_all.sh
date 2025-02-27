#!/bin/bash


save_dir_GraphProp="./gpp_exp/" # in this folder are stored the data and the results 
gpus=0.5 # the config requires at least 50% of a free gpu to be scheduled
cpus=5 # the config requires at least 5 free cpus to be scheduled

model=SWAN

task=          # TODO: choose between sssp, ecc, and diam
export CUDA_VISIBLE_DEVICES=0
nohup python3 -u main.py --cpus $cpus --gpus $gpus --task $task --model_name $model --save_dir $save_dir_GraphProp >$save_dir_GraphProp/out_$model\_$task  2>$save_dir_GraphProp/err_$model\_$task &
