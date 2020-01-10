#!/bin/bash

project=(AMZ)
step=(verification)
experiment=(3_comparison)
data=(bands)
MODELS=(RF)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2)
ssize=10
trials=2
TRAIN_YEAR='20012002'
YEARS=(2002)
psize_eval=(384)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"

        for year in ${YEARS[@]}; do

            if [ "$reference" = "MCD12Q1v6raw_LCProp2" ]; then
                 num_classes=(11)
            fi

            echo "Evaluating over year: $year and model: $model"
            logfname="E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/$year.log"
            python evaluation.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}" \
                --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize_eval}/multiple" \
                --storedir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${tyear}_${reference}/$year" \
                --classifier $model \
                --writetiles \
                --batchsize=1 \
                --dataset=$year \
                --experiment=$data \
                --step=$step \
                --ref $reference > $logfname 2>&1

        done
    done
done