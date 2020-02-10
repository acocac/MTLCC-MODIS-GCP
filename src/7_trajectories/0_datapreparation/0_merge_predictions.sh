#!/bin/bash

project=(AMZ)
YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009)
nworkers=(0)

trainon=(010203)
epochs=(ep30)
project=(AMZ)
psize=(384)
cell=(128)
experiment=(4_local)
ckp=(169921)
MODELS=convgru
SPLITS=(0)
batchsize=(24)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
optimizertype=(adam)
input=(bands)

logdir="E:/acocac/research/${project}/prediction/_logs/"
mkdir -p $logdir

for reference in ${REFERENCES[@]}; do
    for model in ${MODELS[@]}; do
        for split in ${SPLITS[@]}; do
            for year in ${YEARS[@]}; do

                echo "Merging LC predictions for year: $year"
                logfname="${logdir}/$year.log"

                python 0_merge_predictions.py \
                        --indir="F:/acoca/research/gee/dataset/${project}/eval/pred/$experiment/${epochs}/${model}/${model}${cell}_${trainon}_${optimizertype}_fold${split}_${reference}_${ckp}/$year" \
                        --outdir="F:/acoca/research/gee/dataset/${project}/prediction" \
                        --targetyear $year > $logfname 2>&1
            done
         done
    done
done