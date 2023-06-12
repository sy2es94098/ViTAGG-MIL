#!/bin/bash

base=1
MODEL='vit'
WEIGHT_DIR='/data3/ian/dsmil-wsi/dsmil-wsi/weights/'
DATE='01112023'
STORE_DIR='vit-result/'

DATE_DIR=${STORE_DIR}${DATE}
if [  ! -d ${DATE_DIR} ]; then
	mkdir ${DATE_DIR}
fi


COUNT=$(ls ${WEIGHT_DIR}${DATE} | wc -l)

echo ${COUNT}

for ((i=1;i<=${COUNT};i++)); do
	WEIGHT=${WEIGHT_DIR}${DATE}'/'${i}'.pth'

	RESULT_DIR=${DATE_DIR}'/'${i}
	mkdir ${RESULT_DIR}

        python vis_attn_map.py --aggr_weight ${WEIGHT} --store_dir ${RESULT_DIR} --model ${MODEL}

done
