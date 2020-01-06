
#!/bin/bash

project=(AMZ)
step=(eval)
tyear=(15)
experiment=(4_comparison)
MODELS=(RF)
REFERENCES=(MCD12Q1v6raw_LCType1)
trainsize=1000
testsize=1000
channels=(8)
YEARS=(2015)
psize_eval=(384)


for reference in ${REFERENCES[@]}; do

    for model in ${MODELS[@]}; do

        mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_train${trainsize}test${testsize}_${tyear}_${reference}"

        for year in ${YEARS[@]}; do

            if [ "$reference" = "MCD12Q1v6raw_LCType1" ]; then
                 num_classes=(17)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp1" ]; then
                 num_classes=(16)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp2" ]; then
                 num_classes=(11)
            elif [ "$reference" = "ESAraw" ]; then
                 num_classes=(37)
            elif [ "$reference" = "Copernicusraw" ]; then
                 num_classes=(22)
            elif [ "$reference" = "Copernicusnew_all" ]; then
                 num_classes=(22)
            elif [ "$reference" = "Copernicusnew_cebf" ]; then
                 num_classes=(22)
            fi

            echo "Processing year: $year and model: $model"
            logfname="E:/acocac/research/${project}/eval/pred/$experiment/_logs/${model}_train${trainsize}test${testsize}_${tyear}_${reference}/$year.log"
            python 1_evaluate_RF.py "E:/acocac/research/${project}/models/$experiment/${model}_train${trainsize}test${testsize}_${tyear}_${reference}_8d" \
                --datadir="F:/acoca/research/gee/dataset/${project}/MOD09_250m500m/gz/${psize_eval}/multiple" \
                --storedir="E:/acocac/research/${project}/eval/pred/$experiment/${model}/${model}_train${trainsize}test${testsize}_${tyear}_${reference}/$year" \
                --writetiles \
                --batchsize=1 \
                --dataset=$year \
                --experiment=$step \
                --num_classes $num_classes \
                --ref $reference > $logfname 2>&1

        done
    done
done
 #           --writeconfidences \
 #       mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/${model}/${model}_train${trainsize}test${testsize}_${year}_${reference}"
