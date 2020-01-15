#!/bin/bash

project=(AMZ)
TRAIN_YEAR='2001|2002|2003'
train_year='200120022003'
experiment=(3_comparison)
MODELS=(SVM)
ssize=(3000)
cpus=(12)
FOLDS=(1 2 3 4)
trials=(30)

for model in ${MODELS[@]}; do
    for fold in ${FOLDS[@]}; do
        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Training the best model of $model with fold $fold"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}_run_fold${fold}.log"
        python train.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${train_year}" \
            --classifier=$model \
            --datadir="F:/acoca/research/gee/dataset/${project}/comparison/input/all/raw" \
            --train_on "${TRAIN_YEAR}" \
            --ssize $ssize \
            --fold $fold \
            --trials $trials \
            --cpus $cpus > $logfname 2>&1
    done
done