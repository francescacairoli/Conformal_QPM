#!/bin/bash

############
# settings #
############

MODEL_PREFIX="EHT" 
MODEL_DIM=2
echo $MODEL_PREFIX
echo $MODEL_DIM

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

echo $OUT

#######
# run #
#######
#0 1 2 3 4 5 6 7
for i in -1
do
	echo i: $i
	python exec_qpm.py --qr_training_flag False --n_epochs 500 --property_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM --type_localizer knn --knn 10 >> $OUT 2>&1
	#python exec_qpm.py --qr_training_flag False --n_epochs 500 --property_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM --type_localizer gauss >> $OUT 2>&1

done