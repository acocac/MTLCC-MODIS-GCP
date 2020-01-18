#!/bin/bash

project=(AMZ)
experiment=(3_comparison)
MODELS=(RF)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
ssize=3000
trials=100
TRAIN_YEAR='200120022003'
YEARS=(2001)
#FOLDS=(0 1 2 3 4 5)
FOLDS=(0)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"

        for fold in ${FOLDS[@]}; do

            for year in ${YEARS[@]}; do

                echo "Evaluating over year: $year and model: $model and fold: $fold"
                logfname="E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/${year}_fold${fold}_in.log" \

                python 0_genNPY.py \
                    --verdir="E:/acocac/research/${project}/eval/verification/$reference/$year" \
                    --preddir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}/fold$fold/$year" \
                    --fold $fold \
                    --dataset=$year > $logfname 2>&1

            done
        done
    done
done