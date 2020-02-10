#!/bin/bash

project=(AMZ)
blocks=1
psize=384
year=2005
REFERENCES=(Copernicusraw)

for reference in ${REFERENCES[@]}; do
        mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs/predict/"
        echo "Processing reference $reference"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/predict/$reference.log"
        python 5_assign_partition_GZfiles_predict_work.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --tyear=$year \
            --psize=$psize  \
            --blocks=$blocks \
            --reference=$reference > $logfname 2>&1
done
