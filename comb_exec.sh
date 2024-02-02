#!/bin/bash

############
# settings #
############

# model dim = 2 nb comb = 1
# model dim = 4 nb comb = 6
# model dim = 8 nb comb = 28
MODEL_PREFIX="MRH" 
MODEL_DIM=8 

echo $MODEL_PREFIX
echo $MODEL_DIM

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="out/logs/"
mkdir -p $LOGS
OUT="${LOGS}COMB_${MODEL_PREFIX}${MODEL_DIM}_${DATE}_${TIME}_out.txt"

#######
# run #
#######
#0 1 2 3 4 5 
for i in 0 1 2 3 4 5 6
do
	echo i: $i
	python exec_comb_qpm.py --qr_training_flag True --comb_calibr_flag True --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
	python exec_comb_qpm.py --qr_training_flag True --comb_calibr_flag False --comb_idx $i --model_prefix $MODEL_PREFIX --model_dim $MODEL_DIM >> $OUT 2>&1
done