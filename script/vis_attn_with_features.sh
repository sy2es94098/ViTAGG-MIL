#!/bin/bash

GPU='2'
MODEL='vit'
DATASET='C16_ms_testset_fusion'
PATCHES='test_patches/patches/'
BG_DIR='test_patches/patches/'
#PATCHES='test_patches/split/'
#PATCHES='test_patches/subset'
#THRESH=0.9761684536933899
THRESH=0.05
N_CLS=1
B_LOSS=0.9
#AGGR_WEIGHT='test-c16-vit-train/weights/aggregator.pth'
#AGGR_WEIGHT=weight_vit/01272023/1_9_.pth
#AGGR_WEIGHT=weight_vit/01282023/4_6.pth #0.5 0.5
#AGGR_WEIGHT=weight_vit/01292023/1_24.pth
#AGGR_WEIGHT=weight_vit/01312023/1_24.pth
#AGGR_WEIGHT=weight_vit/02012023/1_24.pth #* 1 0
#AGGR_WEIGHT=weight_vit/02012023/8_27.pth
#AGGR_WEIGHT=weight_vit/02032023/1_44.pth #0.5 0.5
#AGGR_WEIGHT=weight_vit/02032023/11_27.pth #1 0
#AGGR_WEIGHT=weight_vit/02042023/1_14.pth # 0.1 0.9
#AGGR_WEIGHT=weight_tcga/02092023/1_3.pth
#------------------------------
#AGGR_WEIGHT=weight_c16/02172023/1_14.pth #abalation
#------------------------------------
#AGGR_WEIGHT=weight_c16/02262023/0.001_0.8_False_1_9.pth
#AGGR_WEIGHT=weight_c16/02222023/0.00008_0_False_5_9.pth
#AGGR_WEIGHT=weight_c16/02262023/0.001_1_False_1_32.pth
#AGGR_WEIGHT=weight_c16/02242023/0.00008_0.2_False_4_13.pth
#AGGR_WEIGHT=test-c16/weights/aggregator.pth
DATE=05172023
WEIGHT_PATH='LR_0.002_BLOSS_0.9_POS_0_HEAD_2_BLOCK_2_LANDMARK_2048_ms_fusion_1_12'
AGGR_WEIGHT='weight_c16_ms/'$DATE'/'$WEIGHT_PATH'.pth'
mkdir $DATE

POSTFIX=$DATASET'_'$WEIGHT_PATH
STORE_DIR=$DATE'/'$POSTFIX
SIZE=512

HEAD=2
BLOCK=2
LANDMARK=2048

export CUDA_VISIBLE_DEVICES=${GPU}
mkdir ${STORE_DIR}
echo 'store at '$STORE_DIR
python vis_attn_map_with_feature.py --aggr_weight ${AGGR_WEIGHT} --store_dir ${STORE_DIR} \
			--model ${MODEL} --thres_tumor ${THRESH} \
			--num_classes ${N_CLS}  \
			--postfix ${POSTFIX} --b_loss ${B_LOSS} \
			--feats_size ${SIZE} --dataset ${DATASET} \
			--bg_dir ${BG_DIR} \
            		--num_head $HEAD --num_block $BLOCK --num_landmark $LANDMARK
