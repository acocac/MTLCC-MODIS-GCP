#!/bin/bash

project=(tile_0_563)
experiment=(2_gcloud)
MODELS=(CE)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
ssize=500
trials=100
TRAIN_YEAR='200120022003'
YEARS=(2001)
level=(global)
folds=(0)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"

        for year in ${YEARS[@]}; do

            echo "Evaluating over year: $year and model: $model and level $level"
            logfname="E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/${year}_out_${level}.log"\

            python 1_metrics.py \
                --indir="E:/acocac/research/${project}/eval/metrics/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}" \
                --outdir="E:/acocac/research/${project}/eval/metrics/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}/output" \
                --dataset=$reference \
                --targetyear=$year \
                --folds=$folds \
                --level=$level > $logfname 2>&1
        done
    done
done