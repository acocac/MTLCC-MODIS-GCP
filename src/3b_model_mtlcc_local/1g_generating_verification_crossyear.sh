
#!/bin/bash

#YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017)
YEARS=(2015)
PROJECTS=(tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713 tile_1_463)
#PROJECTS=(tile_0_201)
psize=(384)
experiment=(1_dataset)
REFERENCES=(Copernicusnew_cf2others merge_datasets2own MCD12Q1v6stable_LCType1) #mapbiomas_fraction watermask [only 2018]

input=(bands)

for reference in ${REFERENCES[@]}; do
    for project in ${PROJECTS[@]}; do
        echo $reference
        mkdir -p "E:/acocac/research/${project}/eval/verification"

        if [ "$reference" = "mapbiomas_fraction" ]; then
            foldername=("mapbiomas")
        elif [ "$model" = "bands250m" ]; then
            foldername=("watermask")
        else
            foldername=$reference
        fi

        mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername"
        mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername/_logs"

        for year in ${YEARS[@]}; do
            echo "Processing project: $project and year: $year and reference: $reference"

            mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername/$year"

            logfname="E:/acocac/research/${project}/eval/verification/$foldername/_logs/$year.log"
            python verification.py \
                --datadir="F:/acoca/research/gee/dataset/${project}/gz/${psize}/multiple" \
                --storedir="E:/acocac/research/${project}/eval/verification/$foldername/$year" \
                --writetiles \
                --batchsize=1 \
                --dataset=$year \
                --experiment='bands' \
                --reference=$reference \
                --step='verification' \
                --allow_growth TRUE > $logfname 2>&1
        done
    done
done