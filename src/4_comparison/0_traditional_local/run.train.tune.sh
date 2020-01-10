#!/bin/bash

project=(AMZ)
dataset=(all)
TRAIN_YEAR='2001|2002|2003'
experiment=(3_comparison)
MODELS=(RF)
ssize=(3000)
channels=(7)
partition=(val)
cpus=(-1)
num_eval=(100)
reference=(MCD12Q1v6stable01to03_LCProp2)

for model in ${MODELS[@]}; do
        mkdir -p "E:/acocac/research/${project}/models/$experiment/_logs"
        echo "Processing $model and train-on $TRAIN_YEAR and N trials $num_eval"
        logfname="E:/acocac/research/${project}/models/$experiment/_logs/${model}_ssize${ssize}_trials${num_eval}_trainon${TRAIN_YEAR}_${reference}.log"
        python train-hpt.py "E:/acocac/research/${project}/models/$experiment" \
            --classifier=$model \
            --datadir="F:/acoca/research/gee/dataset/${project}/comparison/input/$dataset/raw" \
            --train_on "${TRAIN_YEAR}" \
            --ssize $ssize \
            --partition $partition \
            --cpus $cpus \
            --num_eval $num_eval \
            --writemodel \
            --channels=$channels > $logfname 2>&1
done