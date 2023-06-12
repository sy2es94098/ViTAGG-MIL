#!/bin/bash
DIR='/data3/ian/dsmil-wsi/dsmil-wsi/full_test_location/percentile/'
for i in {2..9}
do
	NUM=$((i*10))
	echo $NUM
	STORE=${DIR}${i}
	echo  ${STORE}
	mkdir ${STORE}
	python attention_threshold_percentile.py ${STORE} ${NUM}
	python fill_mask.py ${STORE}
done
