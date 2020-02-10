
#!/bin/bash

PROJECTS=(tile_1_463 tile_0_201 tile_0_143 tile_0_365 tile_1_438 tile_0_630 tile_1_713)
psize=(384)
experiment=(1_dataset)
syear=(2001)
REFERENCES=(mapbiomas)

for project in ${PROJECTS[@]}; do
    for reference in ${REFERENCES[@]}; do

        mkdir -p "E:/acocac/research/${project}/post/_logs/$experiment/verification"
        mkdir -p "E:/acocac/research/${project}/eval/verification"

        logfname="E:/acocac/research/${project}/post/_logs/$experiment/verification/${reference}.log"

        echo "Processing project: $project and dataset: $reference"

        python 1_mapbiomas_tile.py \
                    --indir "E:/acocac/research/${project}/eval/verification/${reference}" \
                    --outdir="E:/acocac/research/${project}/post" \
                    --tile=$project \
                    --psize=$psize \
                    --dataset=$reference \
                    --startyear=$syear \
                    --terrai TRUE \
                    --experiment=experiment > $logfname 2>&1
    done
done






