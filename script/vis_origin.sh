#!/bin/bash

#BASE=6
BASE=1
MODEL='origin'
#WEIGHT_DIR='/data3/ian/dsmil-wsi/dsmil-wsi/weights_origin/'
WEIGHT_DIR='/data3/ian/dsmil-wsi/dsmil-wsi/weights_origin_no_init/'
#DATE='01132023'
DATE='01152023'
STORE_DIR='origin-result-no-init/'

DATE_DIR=${STORE_DIR}${DATE}
if [  ! -d ${DATE_DIR} ]; then
	mkdir ${DATE_DIR}
fi


COUNT=$(ls ${WEIGHT_DIR}${DATE} | wc -l)

echo ${COUNT}

for ((i=${BASE};i<=${COUNT};i++)); do
	WEIGHT=${WEIGHT_DIR}${DATE}'/'${i}'.pth'

	RESULT_DIR=${DATE_DIR}'/'${i}
	mkdir ${RESULT_DIR}

        python vis_attn_map_origin.py --aggr_weight ${WEIGHT} --store_dir ${RESULT_DIR} --model ${MODEL}

done
