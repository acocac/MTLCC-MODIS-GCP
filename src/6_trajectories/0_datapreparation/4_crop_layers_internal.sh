#!/bin/bash

project=(AMZ)
nworkers=(0)

logdir="F:/acoca/research/gee/dataset/${project}/aux/_logs/"
mkdir -p $logdir

echo "Creating subset AUX data"
logfname="${logdir}/crop_aux.log"

python 4_crop_layers.py \
        --indir="F:/acoca/research/gee/dataset/${project}/implementation/ancillary/gee" \
        --outdir="F:/acoca/research/gee/dataset/${project}/prediction/ep15/aux_subset" \
        --nworkers $nworkers \
    	--noconfirm > $logfname 2>&1