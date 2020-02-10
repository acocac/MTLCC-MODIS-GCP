
#!/bin/bash

model=(bands)
SPLITS=(0)
YEARS=(2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017)
train=(02)
epochs=(ep5)
project=(fc2)
psize=(384)
cell=(convgru64)
experiment=(2_inputs)
ckp=(19713)

for split in ${SPLITS[@]}; do
    mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}"
    mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/${model}"

    for year in ${YEARS[@]}; do
        echo "Processing year: $year and model: $model"
        logfname="E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}/$year.log"
        python evaluate.py "E:/acocac/research/${project}/eval/models/$experiment/${epochs}/${cell}_p${psize}pxk0px_batch32_${train}_MCD12Q1v6_cleaned_8d_l1_bidir_${model}_fold${split}_ckp${ckp}" \
            --datadir="F:/acoca/research/gee/dataset/${project}/MOD09_250m500m/gz/${psize}/MCD12Q1v6" \
            --storedir="E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/${model}/${cell}_${train}_fold${split}_${ckp}/$year" \
            --writetiles \
            --writeconfidences \
            --batchsize=1 \
            --dataset=$year \
            --experiment=$model \
            --writegt \
        --allow_growth TRUE > $logfname 2>&1

    done
done