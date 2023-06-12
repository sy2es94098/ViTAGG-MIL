#!/bin/bash
GPU='0'
#DATASET='Camelyon16'
#DATASET='TCGA_train'
#DATASET='Camelyon16-20x-5x-cat'
DATASET='C16_ms_training'

STORE='weight_c16_ms'
#STORE='weight_tcga'

N_CLS=1
EP=40
#LR=0.002
SIGMOID=False
B_LOSS=1
SPLIT=0.2
WITH_POS=0
SIZE=1024
DATE='05012023'
HEAD=6
BLOCK=2

for i in {13..15}
do
	#LR=$(bc -l <<< $i/2500)
	LR=$(bc -l <<< 'scale=5; '$i'/5000')
	#echo $LR
	PREFIX=${WITH_POS}'_'${LR}'_'${B_LOSS}'_'${SIGMOID}

	python train_tcga.py --gpu_index ${GPU} --dataset ${DATASET} \
	        	     --store_dir ${STORE} --num_epochs ${EP} --lr ${LR} --num_classes ${N_CLS} \
		             --sigmoid ${SIGMOID} --prefix ${PREFIX} --b_loss ${B_LOSS} --split ${SPLIT} \
			     --with_pos ${WITH_POS}  --feats_size ${SIZE} \
       			     --num_head $HEAD --num_block $BLOCK	 	
			     #| tee $DATE'_''train_'$PREFIX'.log'

done
