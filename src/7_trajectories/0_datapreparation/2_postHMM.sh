#!/bin/bash

project=(AMZ)

logdir="F:/acoca/research/gee/dataset/${project}/prediction/_logs/"
mkdir -p $logdir

echo "Creating LC HMM maps"
logfname="${logdir}/predicthmm.log"

python 2_postHMM.py \
    --indir="F:/acoca/research/gee/dataset/${project}/prediction/ep15/confidences_stack_subset" \
    --outdir="F:/acoca/research/gee/dataset/${project}/prediction/ep15/prediction_hmm_subset" > $logfname 2>&1