#!/bin/bash

project=(tl_bogota)
blocks=16
psize=384
year=2001
REFERENCES=(Copernicusraw)

for reference in ${REFERENCES[@]}; do
        mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs/predict/"
        echo "Processing reference $reference"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/predict/$reference.log"
        python 3_assign_partition_GZfiles_train.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --tyear=$year \
            --psize=$psize  \
            --blocks=$blocks \
            --fold=0 \
            --reference=$reference > $logfname 2>&1
done
