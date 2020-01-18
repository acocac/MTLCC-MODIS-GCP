#!/bin/bash

project=(AMZ)
step=(verification)
experiment=(3_comparison)
data=(bands)
MODELS=(RF)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
ssize=3000
trials=100
TRAIN_YEAR='200120022003'
YEARS=(2001 2002 2003)
psize_eval=(384)
FOLDS=(0)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"

        for fold in ${FOLDS[@]}; do

            for year in ${YEARS[@]}; do

                if [ "$reference" = "MCD12Q1v6raw_LCProp2" ]; then
                     num_classes=(11)
                fi

                echo "Evaluating over year: $year and model: $model and fold: $fold and ssize: $ssize"
                logfname="E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/fold${fold}_${year}.log"
                python evaluation.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_$reference/models" \
                    --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize_eval}/multiple" \
                    --storedir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}/fold${fold}/$year" \
                    --classifier $model \
                    --writetiles \
                    --batchsize=1 \
                    --fold=$fold \
                    --dataset=$year \
                    --experiment=$data \
                    --step=$step \
                    --reference $reference \
                    --ref $reference > $logfname 2>&1
            done
        done
    done
done