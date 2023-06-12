#!/bin/bash

GPU='0'
MODEL='vit'
PATCHES='test_patches/patches/'
#PATCHES='test_patches/split/'

#THRESH=0.9761684536933899
THRESH=0.05
N_CLS=1
POST_FIX='with_pos_6'
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
DATE=04192023
AGGR_WEIGHT=weight_c16/$DATA/
export CUDA_VISIBLE_DEVICES=${GPU}
mkdir ${DATE}

for i in {1..5}
do
	#LR=$(bc -l <<< $i/2500)
	LR=$(bc -l <<< 'scale=5; '$i'/2500')
	#echo $LR
	PREFIX=${WITH_POS}'_'${LR}'_'${B_LOSS}'_'${SIGMOID}
	python vis_attn_map.py --aggr_weight ${AGGR_WEIGHT} --store_dir ${STORE_DIR} \
			       --model ${MODEL} --thres_tumor ${THRESH} \
			       --num_classes ${N_CLS} --patch ${PATCHES} \
			       --post_fix ${POST_FIX} --b_loss ${B_LOSS}
	
done
					
