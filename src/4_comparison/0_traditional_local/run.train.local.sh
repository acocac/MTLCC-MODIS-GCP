#!/bin/bash

project=(AMZ)
TRAIN_YEAR='2001|2002|2003'
train_year='200120022003'
experiment=(3_comparison)
MODELS=(RF)
ssize=(500)
cpus=(4)
FOLDS=(1 2 3 4)
trials=(100)
reference=(MCD12Q1v6stable01to03_LCProp2_major)

for model in ${MODELS[@]}; do
    for fold in ${FOLDS[@]}; do
        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Training the best model of $model with fold $fold"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}_run_fold${fold}.log"
        python train.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${train_year}_$reference" \
            --classifier=$model \
            --datadir="F:/acoca/research/gee/dataset/${project}/comparison/input" \
            --train_on "${TRAIN_YEAR}" \
            --ssize $ssize \
            --fold $fold \
            --reference $reference \
            --trials $trials \
            --cpus $cpus > $logfname 2>&1
    done
done