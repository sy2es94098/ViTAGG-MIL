#!/bin/bash
GPU='2'
#DATASET='Camelyon16'
#DATASET='Camelyon16-20x-5x-cat'
#DATASET='TCGA_train'
DATASET='C16_ms_training_fusion_with_pos'
STORE='weight_c16_ms_pseudo_bag'
#STORE='weight_tcga'
N_CLS=1
EP=100
LR=0.0004
SIGMOID=False
B_LOSS=1
SPLIT=0.2
WITH_POS=0
SIZE=512
HEAD=4
BLOCK=1
LANDMARK=512
PREFIX='LR_'${LR}'_BLOSS_'${B_LOSS}'_POS_'${WITH_POS}'_HEAD_'${HEAD}'_BLOCK_'$BLOCK'_LANDMARK_'$LANDMARK'_ms_fusion'

python train_with_pseudo_bag.py --gpu_index ${GPU} --dataset ${DATASET} \
	--store_dir ${STORE} --num_epochs ${EP} --lr ${LR} --num_classes ${N_CLS} \
	--sigmoid ${SIGMOID} --prefix ${PREFIX} --b_loss ${B_LOSS} --split ${SPLIT} \
	--with_pos ${WITH_POS} --feats_size ${SIZE} \
	--num_head $HEAD --num_block $BLOCK --num_landmark $LANDMARK

