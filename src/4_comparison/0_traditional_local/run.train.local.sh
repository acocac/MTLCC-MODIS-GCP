#!/bin/bash

project=(AMZ)
TRAIN_YEAR='2001|2002|2003'
experiment=(4_comparison)
MODELS=(RF)
ssize=(3000)
cpus=(12)

for model in ${MODELS[@]}; do
        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Training the best model of $model"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}.log"
        python train.py "E:/acocac/research/${project}/models/$experiment/${model}_ssize${ssize}_${TRAIN_YEAR}" \
            --classifier=$model \
            --datadir="F:/acoca/research/gee/dataset/${project}/comparison/input/all/interpolated" \
            --train_on "${TRAIN_YEAR}" \
            --ssize $ssize \
            --trials "F:\acoca\research\gee\dataset\AMZ\comparison\output\sample\interpolated\hyperopt_trials_niters100_ssize3000.pkl" \
            --cpus $cpus > $logfname 2>&1
done