#!/bin/bash

YEARS=(2002 2003)
PROJECTS=(AMZ)
psize=(384)
maxblocks=(1)
partition=(eval) #train_forest70 eval crossyear

mkdir -p "F:/acoca/research/gee/dataset/$project/_logs"

if [ "$partition" = 'train_forest70' ] || [ "$partition" = "train" ];  then
    nworkers=(12)
elif [ "$partition" = 'eval' ] || [ "$partition" = "crossyear" ];  then
    nworkers=(12)
fi

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
        --nworkers=$nworkers \
    	--noconfirm > $logfname 2>&1
    done
done