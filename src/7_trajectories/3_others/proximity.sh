#!/bin/bash

project=(AMZ)
TVALUES=(6)

logdir="F:/acoca/research/gee/dataset/${project}/implementation/_logs/"
mkdir -p $logdir

logfname="${logdir}/proximity.log"

for tvalue in ${TVALUES[@]}; do
    python gdal_proximity2.py \
            -i "F:/acoca/research/gee/dataset/${project}/implementation/ancillary/2009_tmp.tif" \
            -o "F:/acoca/research/gee/dataset/${project}/implementation/ancillary/local/lc/2019/dc${tvalue}.tif" \
            -t $tvalue > $logfname 2>&1
done