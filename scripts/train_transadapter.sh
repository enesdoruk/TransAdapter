#! /usr/bin/env bash


export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=12355


CUDA_VISIBLE_DEVICES=1 python train_transadapter.py \
                            --train_batch_size 11 \
                            --eval_batch_size 11 \
                            --dataset office \
                            --source_list data/office/webcam_list.txt \
                            --target_list data/office/amazon_list.txt \
                            --test_list_source data/office/webcam_list.txt  \
                            --test_list_target data/office/amazon_list.txt \
                            --name aw \
                            --local_rank -1 \
                            --num_steps 16000 \
                            --learning_rate 5e-2 \
                            --num_classes 31 \
                            --eval_every 200 \
                            --wandb_name transadapter \
                            --pretrained_dir /AI/transAdapter/checkpoints/swin_base_patch4_window7_224_22k.pth \
                            --psedo_ckpt /AI/transAdapter/checkpoints/aw_checkpoint.bin \
                            --gamma 0.001 \
                            --beta 0.1 \
                            --theta 0.0000 \
                            --pseudo_lab True
                            # --sweep False \
