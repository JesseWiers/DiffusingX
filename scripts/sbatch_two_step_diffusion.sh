#!/bin/bash

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --dropout 0.1 --image_size 256 --learn_sigma True --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 24 --microbatch 4"

python scripts/image_train.py --data_dir "datasets/a-large-scale-fish-dataset" --save_dir "trained_models" --in_out_channels 3 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

DATASET=fishes_2

python train_interpreter.py --exp experiments/${DATASET}/ddpm.json $MODEL_FLAGS