#!/bin/bash

project=(AMZ)
folds=(0)
blocks=1
psize=384
year=2015
dataset=(multiple)
REFERENCES=(Copernicusnew_all2ofg)

for reference in ${REFERENCES[@]}; do
    for fold in ${folds[@]}; do
        mkdir -p "F:/acoca/research/gee/dataset/${project}/MOD09_250m500m/_logs/folds/"
        echo "Processing fold $fold and reference $reference"
        logfname="F:/acoca/research/gee/dataset/${project}/MOD09_250m500m/_logs/folds/${fold}_$reference.log"
        python _assign_partition_GZfiles_eval.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}/MOD09_250m500m" \
            --tyear=$year \
            --psize=$psize  \
            --blocks=$blocks \
            --fold=$fold \
            --reference=$reference > $logfname 2>&1
    done
done
