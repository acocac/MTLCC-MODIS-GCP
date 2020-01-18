
#!/bin/bash

YEARS=(2001)
PROJECTS=(AMZ)
samplingsize=(3000)
PARTITIONS=(train)
FOLDS=(4)
storage=(GCloud)
bucket=('thesis-2019')
reference='MCD12Q1v6stable01to03_LCProp2_major'

for project in ${PROJECTS[@]}; do

    mkdir -p "F:/acoca/research/gee/dataset/$project/comparison/input/_logs"

    for fold in ${FOLDS[@]}; do

        outdir="$project/comparison/fold$fold"

        for year in ${YEARS[@]}; do

                for partition in ${PARTITIONS[@]}; do

                logfname="F:/acoca/research/gee/dataset/${project}/comparison/input/_logs/${year}_${partition}_fold${fold}.log"

                echo "Downloading data for project $project and year $year and sampling size of $samplingsize for partition $partition with fold $fold"

                    python 1_query_data_traditional.py \
                        -o=$project \
                        -r=$reference \
                        -y=$year \
                        -p=$partition \
                        -i=$samplingsize \
                        -f=$fold \
                        -s=$storage \
                        -b=$bucket > $logfname 2>&1
            done
        done
    done
done