#!/bin/bash

DIR='/data3/ian/dsmil-wsi/dsmil-wsi/full_test_location/binary_output/'
for i in {10..20}
do
	STORE=${DIR}${i}
	echo  ${STORE}
	mkdir ${STORE}
	python attention_threshold.py ${STORE} ${i}
	python fill_mask.py ${STORE}

done
