#!/bin/bash

project=(AMZ)
years="2004 2017"
size=4

logdir="F:/acoca/research/gee/dataset/${project}/implementation/_logs/"
mkdir -p $logdir

echo "Creating sieve size $size for Terra-i's year(s): $years"
logfname="${logdir}/terrai_sieve.log"

python 1_prepare_sieve_terrai.py \
        --indir="F:/acoca/research/gee/dataset/${project}/implementation" \
        --outdir="F:/acoca/research/gee/dataset/${project}/implementation/terrai_sieve" \
        --targetyear $years \
        --size $size 2>&1