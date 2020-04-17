#!/bin/bash

project=(AMZ)
TRAIN_YEAR='2001|2002|2003'
train_year='200120022003'
experiment=(3_comparison)
MODELS=(SVM)
ssize=(3000)
cpus=(-1)
fold=0
trials=(25)
BESTMODELS=(5)
reference=(MCD12Q1v6stable01to15_LCProp2_major)

for model in ${MODELS[@]}; do
    for bestmodel in ${BESTMODELS[@]}; do
        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Training $model with bestmodel $bestmodel"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}_run_bm${bestmodel}.log"
        python train.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_trials${trials}_trainon${train_year}_$reference" \
            --classifier=$model \
            --datadir="F:/acoca/research/gee/dataset/${project}/comparison/input" \
            --train_on "${TRAIN_YEAR}" \
            --ssize $ssize \
            --fold $fold \
            --reference $reference \
            --trials $trials \
            --bestmodel $bestmodel \
            --cpus $cpus > $logfname 2>&1
    done
done
