#!/bin/bash

project=(AMZ)
experiment=(3_comparison)
MODELS=(RF)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
ssize=3000
trials=100
TRAIN_YEAR='200120022003'
YEARS=(2001 2002 2003)
#FOLDS=(0 1 2 3 4 5)
fold=(0)
BESTMODELS=(1 2 3)

for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        for bestmodel in ${BESTMODELS[@]}; do

            for year in ${YEARS[@]}; do

                if [ "$model" = 'RF' ] || [ "$model" = 'SVM' ] ; then
                    mkdir -p "E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}"
                    preddir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_ssize${ssize}_${TRAIN_YEAR}_${reference}/$bestmodel/$year/prediction"
                    logfname="E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}_ssize${ssize}_trials${trials}_trainon${TRAIN_YEAR}_${reference}/${year}_bm${bestmodel}_in.log"
                else
                    mkdir -p "E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}"
                    preddir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/$bestmodel/$year/prediction"
                    logfname="E:/acocac/research/${project}/eval/metrics/$experiment/_logs/${model}/${year}_bm${bestmodel}_in.log"
                fi

                echo "Evaluating over year: $year and model: $model and best model: $bestmodel"
                logfname=$logfname \

                python 0_genNPY.py \
                    --verdir="E:/acocac/research/${project}/eval/verification/$reference/$year" \
                    --preddir=$preddir \
                    --bestmodel $bestmodel \
                    --dataset=$year > $logfname 2>&1

            done
        done
    done
done