#!/usr/bin/env bash
--fp16 --scale 2 --max_frames_per_gpu 3 --load_size 1024 --dataset_mode temporal --openpose_only --load_size 1024 --fine_size 1024 --name pose2body_scale2 --gpu_ids 1 --n_gpus_gen 1 --dataroot /home/lhuo9710/dataset/everybody_dance_now/subject1/train --no_struct  --no_texture --niter_decay 0 --niter 5 --debug