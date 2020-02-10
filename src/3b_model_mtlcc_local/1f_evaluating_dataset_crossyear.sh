
#!/bin/bash

YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
train=(15)
epochs=(ep30)
major_project=(AMZ)
MINOR_projects=(tile_1_463 tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713)
psize=(384)
cell=(64)
experiment=(1_dataset)
batchsize=(32)
SPLITS=(0)
MODELS=(convgru)
ckp=(42497)
#REFERENCES=(Copernicusnew_all Copernicusnew_cebf)
#REFERENCES=(MCD12Q1v6raw_LCProp1 MCD12Q1v6raw_LCType1)
REFERENCES=(MCD12Q1v6stable_LCProp2)

input=(bands)

for minor_project in ${MINOR_projects[@]}; do
    for reference in ${REFERENCES[@]}; do
        for model in ${MODELS[@]}; do
            for split in ${SPLITS[@]}; do

                mkdir -p "E:/acocac/research/${minor_project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${reference}"
                mkdir -p "E:/acocac/research/${minor_project}/eval/pred/$experiment/${epochs}/${model}"

                for year in ${YEARS[@]}; do
                    echo "Processing tile: $minor_project and year: $year and model: $model and split: $split and reference: $reference"
                    logfname="E:/acocac/research/${minor_project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${reference}/$year.log"
                    python evaluate.py "E:/acocac/research/${major_project}/eval/models/$experiment/${epochs}/${model}${cell}_p${psize}pxk0px_batch${batchsize}_${train}_${reference}_8d_l1_bidir_${input}_fold${split}_ckp${ckp}" \
                        --datadir="F:/acoca/research/gee/dataset/${minor_project}/gz/${psize}/multiple" \
                        --storedir="E:/acocac/research/${minor_project}/eval/pred/$experiment/${epochs}/${model}/${model}${cell}_${train}_fold${split}_${reference}_${ckp}/$year" \
                        --writetiles \
                        --batchsize=1 \
                        --dataset=$year \
                        --experiment='bands' \
                        --writeconfidences \
                        --ref $reference \
                        --writegt \
                    --allow_growth TRUE > $logfname 2>&1
                done
            done
        done
    done
done



