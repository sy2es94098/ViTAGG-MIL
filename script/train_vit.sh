#!/bin/bash
GPU='1'
#DATASET='Camelyon16'
#DATASET='Camelyon16-20x-5x-cat'
#DATASET='TCGA_train'
DATASET='C16_ms_training_fusion'
STORE='weight_c16_ms'
#STORE='weight_tcga'
N_CLS=1
EP=100
LR=0.002
SIGMOID=False
B_LOSS=0.9
SPLIT=0.1
WITH_POS=0
SIZE=512
HEAD=2
BLOCK=2
LANDMARK=2048
PREFIX='LR_'${LR}'_BLOSS_'${B_LOSS}'_POS_'${WITH_POS}'_HEAD_'${HEAD}'_BLOCK_'$BLOCK'_LANDMARK_'$LANDMARK'_ms_fusion'

python train_tcga.py --gpu_index ${GPU} --dataset ${DATASET} \
	--store_dir ${STORE} --num_epochs ${EP} --lr ${LR} --num_classes ${N_CLS} \
	--sigmoid ${SIGMOID} --prefix ${PREFIX} --b_loss ${B_LOSS} --split ${SPLIT} \
	--with_pos ${WITH_POS} --feats_size ${SIZE} \
	--num_head $HEAD --num_block $BLOCK --num_landmark $LANDMARK

