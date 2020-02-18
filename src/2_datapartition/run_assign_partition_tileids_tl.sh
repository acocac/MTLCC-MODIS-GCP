#!/bin/bash

project=(tl_bogota)
blocks=1
psize=384
year=2001
REFERENCES=(Copernicusraw)

for reference in ${REFERENCES[@]}; do
        mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs/predict/"
        echo "Processing reference $reference"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/predict/$reference.log"
        python 6_assign_partition_GZfiles_transferlearning.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --tyear=$year \
            --psize=$psize  \
            --blocks=$blocks \
            --reference=$reference > $logfname 2>&1
done
