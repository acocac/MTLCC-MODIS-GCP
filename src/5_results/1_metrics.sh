#!/bin/bash

project=(tl_bogota)
experiment=(0_tl)
MODELS=(convgru)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
ssize=3000
trials=25
TRAIN_YEAR='200120022003'
YEARS=(2001 2002 2003)
level=(global)
#folds=(0)
bestmodels=(012)

rootdir="F:/acoca/research/gee/dataset/${project}/eval/metrics/$experiment"
logdir="F:/acoca/research/gee/dataset/${project}/eval/metrics/$experiment/_logs/"

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        for year in ${YEARS[@]}; do

            if [ "$model" = 'RF' ] || [ "$model" = 'SVM' ] ; then
                mkdir -p "${logdir}/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"
                indir="${rootdir}/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}"
                logfname="${logdir}/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/${year}_out_${level}.log"
            else
                mkdir -p "${logdir}/${model}"
                indir="${rootdir}/$model"
                logfname="${logdir}/${model}/${year}_out_${level}.log"
            fi

            echo "Evaluating over year: $year and model: $model and level $level"

            python 1_metrics.py \
                --indir=$indir \
                --outdir="${indir}/output" \
                --dataset=$reference \
                --targetyear=$year \
                --bestmodels=$bestmodels \
                --level=$level > $logfname 2>&1
        done
    done
done