#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
#: << EOF
seed="77 52"
DATASET=C16_paper
FEATS_SIZE=2048
LR=0.0002
EP=100
GPU_ID='0 1'
#WEIGHT='weights/simsiam_30ep/1.pth'

for i in $seed; do
        python train_tcga_init.py --dataset=${DATASET} --num_classes 1 --feats_size ${FEATS_SIZE} \
                             --lr ${LR} --num_epochs ${EP} --gpu_index ${GPU_ID} --seed ${i}
#                            --aggregator_weights ${WEIGHT}
done

