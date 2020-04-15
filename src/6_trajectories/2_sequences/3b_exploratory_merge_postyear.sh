#!/bin/bash

project=(AMZ)

logdir="E:/acocac/research/${project}/trajectories/geodata/postyear/_logs/"
mkdir -p $logdir

echo "Creating single raster entropy and turbulence"
logfname="${logdir}/entropyturbulence.log"

python 3b_exploratory_merge_postyear.py \
            --geodir="E:/acocac/research/${project}/trajectories/geodata/postyear/masked" \
            --auxdir="F:/acoca/research/gee/dataset/${project}/implementation" \
            --outdir="E:/acocac/research/${project}/trajectories/geodata/postyear/output" > $logfname 2>&1