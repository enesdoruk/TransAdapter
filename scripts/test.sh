
CUDA_VISIBLE_DEVICES=0,1 python test.py \
                            --eval_batch_size 1 \
                            --dataset office \
                            --test_list_target data/office/amazon_list.txt \
                            --num_classes 31 \
                            --name trailer \
                            --local_rank 0 \
                            --pretrained_dir /AI/UVT/output/office/trailer_checkpoint.bin