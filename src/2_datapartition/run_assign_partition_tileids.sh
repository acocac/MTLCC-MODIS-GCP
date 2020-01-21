#!/bin/bash

project=(AMZ)
folds=(0 1 2 3 4)
blocks=16
psize=384
year=2001
dataset=(multiple)
REFERENCES=(Copernicusnew_all2ofg)

for reference in ${REFERENCES[@]}; do
    for fold in ${folds[@]}; do
        mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs/folds/"
        echo "Processing fold $fold and reference $reference"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/folds/${fold}_$reference.log"
        python 3_assign_partition_GZfiles_train.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --tyear=$year \
            --psize=$psize  \
            --blocks=$blocks \
            --fold=$fold \
            --reference=$reference > $logfname 2>&1
    done
done
