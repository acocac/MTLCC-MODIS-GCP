
#!/bin/bash

YEARS=(2001)
#YEARS=(2019)
trainon=(010203)
epochs=(ep30)
project_model=(tl_asia)
project_predict=(tl_asia)
psize=(384)
cell=(128)
experiment=(5_scratch)
ckp=(3843)
MODELS=(convgru)
SPLITS=(0)
batchsize=(24)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
optimizertype=(adam)
input=(bands)
npatches=(6)
step=(prediction) #prediction evaluation

for reference in ${REFERENCES[@]}; do
    for model in ${MODELS[@]}; do
        for split in ${SPLITS[@]}; do
            mkdir -p "F:/acoca/research/gee/dataset/${project_predict}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${trainon}_${optimizertype}_${reference}"
            mkdir -p "F:/acoca/research/gee/dataset/${project_predict}/eval/pred/$experiment/${epochs}/${model}"

            for year in ${YEARS[@]}; do
                echo "Processing year: $year and model: $model and split: $split and reference: $reference"
                logfname="F:/acoca/research/gee/dataset/${project_predict}/eval/pred/$experiment/${epochs}/_logs/${model}_ckp${ckp}_fold${split}_${trainon}_${optimizertype}_${reference}/$year.log"
                python predict.py "E:/acocac/research/${project_model}/eval/models/$experiment/${epochs}/${model}${cell}_p${psize}pxk0px_batch${batchsize}_${trainon}_${optimizertype}_${reference}_8d_l1_bidir_${input}_fold${split}_ckp${ckp}" \
                    --datadir="F:/acoca/research/gee/dataset/${project_predict}/combine/${psize}" \
                    --storedir="F:/acoca/research/gee/dataset/${project_predict}/eval/pred/$experiment/${epochs}/${model}/${model}${cell}_${trainon}_${optimizertype}_fold${split}_${reference}_${ckp}/$year" \
                    --writetiles \
                    --batchsize=1 \
                    --dataset=$year \
                    --experiment='bands' \
                    --ref $reference \
                    --step=$step \
                    --writegt \
                    --npatches $npatches \
                    --allow_growth TRUE > $logfname 2>&1
            done
        done
    done
done

#                    --writeconfidences \