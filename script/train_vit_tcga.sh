#!/bin/bash
GPU='3'
#DATASET='Camelyon16'
#DATASET='Camelyon16-20x-5x-cat'
DATASET='TCGA_ms_train'
#DATASET='TCGA_ms_train'
STORE='weight_tcga_ms'
#STORE='weight_tcga'
N_CLS=1
EP=100
LR=0.00008
SIGMOID=False
B_LOSS=1
SPLIT=0.1
WITH_POS=1
SIZE=512
HEAD=4
BLOCK=1
LANDMARK=1024
PREFIX='LR_'${LR}'_BLOSS_'${B_LOSS}'_POS_'${WITH_POS}'_HEAD_'${HEAD}'_BLOCK_'$BLOCK'_LANDMARK_'$LANDMARK'_ms'

python train_tcga.py --gpu_index ${GPU} --dataset ${DATASET} \
	--store_dir ${STORE} --num_epochs ${EP} --lr ${LR} --num_classes ${N_CLS} \
	--sigmoid ${SIGMOID} --prefix ${PREFIX} --b_loss ${B_LOSS} --split ${SPLIT} \
	--with_pos ${WITH_POS} --feats_size ${SIZE} \
	--num_head $HEAD --num_block $BLOCK --num_landmark $LANDMARK |tee $PREFIX'.log'

