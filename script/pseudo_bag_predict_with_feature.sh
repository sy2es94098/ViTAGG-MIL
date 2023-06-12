#!/bin/bash

GPU='0 1 2'
MODEL='vit'
#PATCHES='test/patches'
DATASET='C16_ms_testset_fusion_with_pos'
#DATASET='TCGA_test'
STORE_DIR='tmp'
#THRESH=0.9761684536933899
THRESH_LUAD=0.05
THRESH_LUSC=0.05
N_CLS=1
B_LOSS=1
SIZE=512

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
export CUDA_VISIBLE_DEVICES=${GPU}
#AGGR_WEIGHT=weight_tcga/02242023/0.001_1_False_split1_45_9.pth
#AGGR_WEIGHT=weight_tcga/05172023/LR_0.005_BLOSS_1_POS_0_HEAD_2_BLOCK_2_LANDMARK_512_ms_fusion_1_3.pth
DATE=06092023
WEIGHT_PATH='LR_0.0004_BLOSS_1_POS_0_HEAD_4_BLOCK_1_LANDMARK_512_ms_fusion_7_50'
D_WEIGHT_PATH='d_classifier_'$WEIGHT_PATH
A_WEIGHT_PATH='milnet_'$WEIGHT_PATH
AGGR_WEIGHT='weight_c16_ms_pseudo_bag/'$DATE'/'$A_WEIGHT_PATH'.pth'
D_CLASSIFIER_WEIGHT='weight_c16_ms_pseudo_bag/'$DATE'/'$D_WEIGHT_PATH'.pth'
mkdir $DATE

WITH_POS=0
HEAD=4
BLOCK=1
LANDMARK=512

POSTFIX=$DATASET'_'$WEIGHT_PATH
echo $POSTFIX
python pseudo_bag_predict_with_feature.py --aggr_weight ${AGGR_WEIGHT} --store_dir ${STORE_DIR} \
			--model ${MODEL} --thres_luad ${THRESH_LUAD} --thres_lusc ${THRESH_LUSC}\
			--num_classes ${N_CLS} --dataset ${DATASET}  --postfix $POSTFIX\
			 --b_loss ${B_LOSS} --feats_size $SIZE \
			--with_pos $WITH_POS --num_head $HEAD --num_block $BLOCK --num_landmark $LANDMARK \
			--d_weight $D_CLASSIFIER_WEIGHT
			#| tee $POSTFIX'.log'

