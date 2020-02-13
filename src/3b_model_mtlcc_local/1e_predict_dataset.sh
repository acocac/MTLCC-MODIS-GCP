
#!/bin/bash

YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019)
#YEARS=(2019)
trainon=(010203)
epochs=(ep30)
project=(AMZ)
psize=(384)
cell=(128)
experiment=(4_local)
ckp=(169921)
MODELS=(convgru)
SPLITS=(0)
batchsize=(24)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
optimizertype=(adam)
input=(bands)

for reference in ${REFERENCES[@]}; do
    for model in ${MODELS[@]}; do
        for split in ${SPLITS[@]}; do
            mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${trainon}_${optimizertype}_${reference}"
            mkdir -p "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/${model}"

            for year in ${YEARS[@]}; do
                echo "Processing year: $year and model: $model and split: $split and reference: $reference"
                logfname="E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${trainon}_${optimizertype}_${reference}/$year.log"
                python predict.py "E:/acocac/research/${project}/eval/models/$experiment/${epochs}/${model}${cell}_p${psize}pxk0px_batch${batchsize}_${trainon}_${optimizertype}_${reference}_8d_l1_bidir_${input}_fold${split}_ckp${ckp}" \
                    --datadir="T:/BACKUPS/BACKUP_PhDAlejandro/${project}/combine/${psize}" \
                    --storedir="F:/acoca/research/gee/dataset/${project}/eval/pred/$experiment/${epochs}/${model}/${model}${cell}_${trainon}_${optimizertype}_fold${split}_${reference}_${ckp}/$year" \
                    --writetiles \
                    --batchsize=1 \
                    --dataset=$year \
                    --experiment='bands' \
                    --ref $reference \
                    --step='prediction' \
                    --writeconfidences \
                    --allow_growth TRUE > $logfname 2>&1
            done
        done
    done
done