#!/bin/bash

MODELS=(bands)
project=(AMZ)
epochs=30
psize=(24)
cell=(128)
year=(010203)
experiment=(4_local)
SPLITS=(0)
batchsize=(24)
trainon='2001 2002 2003'
optimizertype=(adam)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
step=(training)

for split in ${SPLITS[@]}; do
    for reference in ${REFERENCES[@]}; do
        for model in ${MODELS[@]}; do
            mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
            echo "Processing $model and split: $split and reference: $reference and optimiser $optimizertype"
            logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}_fold${split}_${reference}_${optimizertype}.log"
            python train.py "E:/acocac/research/${project}/models/$experiment/${model}/convgru${cell}_p${psize}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" \
                --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize}/multiple" \
                --epochs=$epochs \
                --shuffle TRUE \
                --batchsize=$batchsize \
                --train_on $trainon \
                --experiment=${model} \
                --fold=$split \
                --step=$step \
                --allow_growth TRUE \
                --max_models_to_keep 5 \
                --save_every_n_hours 3 \
                --ref $reference > $logfname 2>&1
        done
    done
done