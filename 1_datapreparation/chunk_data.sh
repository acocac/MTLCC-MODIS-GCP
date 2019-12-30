#!/bin/bash

#YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
YEARS=(2001 2002 2003)
#PROJECTS=(tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713)
PROJECTS=(tile_0_563)
psize=(384)
maxblocks=(1)
partition=(crossyear)

mkdir -p "F:/acoca/research/gee/dataset/$project/_logs"

for project in ${PROJECTS[@]}; do
    for year in ${YEARS[@]}; do
    echo "Processing $year and project $project"
    logfname="F:/acoca/research/gee/dataset/${project}/_logs/$year.log"
    python 1_chunk_TFrecords.py \
        --rootdir="F:/acoca/research/gee/dataset/${project}" \
        --psize=$psize \
        --tyear=$year \
        --maxblocks=$maxblocks \
        --exportblocks=$partition \
        --dataset='multiple' \
    	--noconfirm > $logfname 2>&1
    done
done

#                --nworkers=12 \
