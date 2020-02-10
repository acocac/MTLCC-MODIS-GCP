#!/bin/bash

project=(AMZ)
step=(verification)
experiment=(3_comparison)
data=(bands)
MODELS=(SVM)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
ssize=3000
trials=25
TRAIN_YEAR='200120022003'
YEARS=(2001)
psize_eval=(384tmp)
fold=0
BESTMODELS=(5)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"

        for bestmodel in ${BESTMODELS[@]}; do

            for year in ${YEARS[@]}; do

                echo "Evaluating over year: $year and model: $model and best model: $bestmodel and ssize: $ssize"
                logfname="E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/${bestmodel}_${year}.log"
                python evaluation.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_$reference/models" \
                    --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize_eval}/multiple" \
                    --storedir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}/${bestmodel}/$year" \
                    --classifier $model \
                    --writetiles \
                    --batchsize=1 \
                    --fold=$fold \
                    --bestmodel $bestmodel \
                    --dataset=$year \
                    --experiment=$data \
                    --step=$step \
                    --reference $reference \
                    --ref $reference > $logfname 2>&1
            done
        done
    done
done