#!/bin/bash

GPU='0'
MODEL='vit'
PATCHES='test/patches'
STORE_DIR='tmp'
#THRESH=0.9761684536933899
THRESH_LUAD=0.05
THRESH_LUSC=0.05
N_CLS=1
POST_FIX='best'
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
AGGR_WEIGHT=weight_tcga/02242023/0.08_1_False_24_48.pth
python vis_tcga.py --aggr_weight ${AGGR_WEIGHT} --store_dir ${STORE_DIR} \
			--model ${MODEL} --thres_luad ${THRESH_LUAD} --thres_lusc ${THRESH_LUSC}\
			--num_classes ${N_CLS} --patch ${PATCHES} \
			--post_fix ${POST_FIX}

