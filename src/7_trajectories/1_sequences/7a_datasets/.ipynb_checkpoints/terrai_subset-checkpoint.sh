#!/bin/bash

DATES=(2004_01_01_to_2012_02_02 2011_01_01_to_2013_01_01 2012_01_01_to_2013_11_01 2012_01_01_to_2014_02_02 2012_01_01_to_2014_03_06 2013_01_01_to_2014_08_13 2013_01_01_to_2014_09_14)

for date in ${DATES[@]}; do
    mkdir -p "V:/terra-i/temp/tile3/_logs"

    echo "Processing $date"
    logfname="V:/terra-i/temp/tile3/_logs/${date}.log"
    terrai-toolbox.bat geodata.jgrid.ExtractGrid \
    "Q:/BACKUPS/_39/Ver_Feb2015/h10v08/detection_${date}/probabilities" \
    "V:/terra-i/temp/tile3/mask.asc" \
    200 200 \
    "V:/terra-i/temp/tile3/jgrid/h10v08/detection_${date}"

    mkdir -p "V:/terra-i/temp/tile3/ascii/h10v08/detection_${date}"

    terrai-toolbox.bat geodata.jgrid.Grid2ASC \
    "V:/terra-i/temp/tile3/jgrid/h10v08/detection_${date}" \
    "V:/terra-i/temp/tile3/ascii/h10v08/detection_${date}" \
    "all" > $logfname 2>&1

done