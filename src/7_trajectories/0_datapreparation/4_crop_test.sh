#!/bin/bash

project=(AMZ)
nworkers=(0)

logdir="F:/acoca/research/gee/dataset/${project}/prediction/_logs/"
mkdir -p $logdir

echo "Creating subset LC confidences data"
logfname="${logdir}/crop.log"

python 4_crop_test.py \
        --indir="F:/acoca/research/gee/dataset/${project}/prediction/ep15/confidences_stack" \
        --outdir="F:/acoca/research/gee/dataset/${project}/prediction/ep15/confidences_stack_subset" \
        --nworkers $nworkers \
    	--noconfirm > $logfname 2>&1