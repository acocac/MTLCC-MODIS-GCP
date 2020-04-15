
#!/bin/bash

YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018)
train=(15)
epochs=(ep30)
#PROJECTS=(tile_1_463 tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713)
PROJECTS=(tile_0_201 tile_1_463 tile_0_630)
psize=(384)
cell=(64)
experiment=(1_dataset)
ckp=(42497)
MODELS=(convgru)
SPLITS=(0)
batchsize=(32)
syear=(2001)
#REFERENCES=(Copernicusraw)
#REFERENCES=(MCD12Q1v6raw_LCProp2 MCD12Q1v6raw_LCProp1 MCD12Q1v6raw_LCType1)
#REFERENCES=(Copernicusraw Copernicusnew_cf2others merge_datasets2own ESAraw MCD12Q1v6raw_LCType1 MCD12Q1v6raw_LCProp1 MCD12Q1v6raw_LCProp2)
REFERENCES=(MCD12Q1v6stable_LCProp2 Copernicusnew_cf2others mapbiomas)
#REFERENCES=(MCD12Q1v6raw_LCType1)

for project in ${PROJECTS[@]}; do

    for reference in ${REFERENCES[@]}; do

        for model in ${MODELS[@]}; do

            mkdir -p "E:/acocac/research/${project}/post/_logs/$experiment/${epochs}/${model}_ckp${ckp}_${reference}"

            for split in ${SPLITS[@]}; do
                logfname="E:/acocac/research/${project}/post/_logs/$experiment/${epochs}/${model}_ckp${ckp}_${reference}/fold${split}.log"

                echo "Processing project: $project and dataset: $reference and model: $model and split: $split"

                python 0_postHMM_tile.py \
                    --indir "E:/acocac/research/${project}/eval/pred/$experiment/${epochs}/${model}/${model}${cell}_${train}_fold${split}_${reference}_${ckp}" \
                    --outdir="E:/acocac/research/${project}/post" \
                    --suffix="${model}${cell}_${train}_fold${split}_${reference}_${ckp}" \
                    --tile=$project \
                    --psize=$psize \
                    --dataset=$reference \
                    --startyear=$syear \
                                       --terrai TRUE \
                    --experiment=experiment > $logfname 2>&1
            done
        done
    done
done







