#!/bin/bash

YEARS=(2001)
PROJECTS=(AMZ)
psize=(384)

for project in ${PROJECTS[@]}; do
    mkdir -p "F:/acoca/research/gee/dataset/${project}/_logs"
    for year in ${YEARS[@]}; do

        if [ "$year" = 2001 ]; then
            timesteps=(45)
        elif [ "$year" != 2001  ]; then
            timesteps=(46)
        fi

        echo "Processing data for project $project and $year"
        logfname="F:/acoca/research/gee/dataset/${project}/_logs/$year.log"
        python 0_merge_multisources.py \
            --rootdir="F:/acoca/research/gee/dataset/${project}" \
            --psize=$psize \
            --tyear=$year \
            --timesteps=$timesteps > $logfname 2>&1

    done
done