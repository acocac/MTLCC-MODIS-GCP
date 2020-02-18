
#!/bin/bash

YEARS=(2001)
PROJECTS=(tl_bogota)
psize=(384)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
input=(bands)

for reference in ${REFERENCES[@]}; do
    for project in ${PROJECTS[@]}; do
        echo $reference
        mkdir -p "F:/acoca/research/gee/dataset/${project}/eval/verification"

        if [ "$reference" = "mapbiomas_fraction" ]; then
            foldername=("mapbiomas")
        elif [ "$model" = "bands250m" ]; then
            foldername=("watermask")
        else
            foldername=$reference
        fi

        mkdir -p "F:/acoca/research/gee/dataset/${project}/eval/verification/$foldername"
        mkdir -p "F:/acoca/research/gee/dataset/${project}/eval/verification/$foldername/_logs"

        for year in ${YEARS[@]}; do
            echo "Processing project: $project and year: $year and reference: $reference"

            mkdir -p "F:/acoca/research/gee/dataset/${project}/eval/verification/$foldername/$year"

            logfname="F:/acoca/research/gee/dataset/${project}/eval/verification/$foldername/_logs/$year.log"
            python verification.py \
                --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize}/multiple" \
                --storedir="F:/acoca/research/gee/dataset/${project}/eval/verification/$foldername/$year" \
                --writetiles \
                --batchsize=1 \
                --dataset=$year \
                --experiment='bands' \
                --reference=$reference \
                --step='verification' \
                --allow_growth TRUE > $logfname 2>&1
        done
    done
done