#!/bin/bash

project=(AMZ)

logdir="F:/acoca/research/gee/dataset/${project}/prediction/_logs/"
mkdir -p $logdir

echo "Creating LC HMM maps"
logfname="${logdir}/predicthmm.log"

python 2_postHMM.py \
    --indir="F:/acoca/research/gee/dataset/${project}/prediction/confidences_stack" \
    --outdir="F:/acoca/research/gee/dataset/${project}/prediction/prediction_hmm" > $logfname 2>&1