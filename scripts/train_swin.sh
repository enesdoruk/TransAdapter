#! /usr/bin/env bash


export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=12355



CUDA_VISIBLE_DEVICES=1 python train_swin.py \
                            --train_batch_size 12 \
                            --eval_batch_size 12 \
                            --dataset office \
                            --source_list data/office/webcam_list.txt \
                            --target_list data/office/amazon_list.txt \
                            --test_list_source data/office/webcam_list.txt  \
                            --test_list_target data/office/amazon_list.txt \
                            --name aw \
                            --local_rank -1 \
                            --num_steps 15000 \
                            --learning_rate 1e-2 \
                            --num_classes 31 \
                            --eval_every 200 \
                            --wandb_name transadapter \
                            --pretrained_dir /AI/UVT/checkpoints/swin_base_patch4_window7_224_22k.pth \
