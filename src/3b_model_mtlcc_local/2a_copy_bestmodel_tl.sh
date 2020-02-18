
#!/bin/bash

MODELS=(bands)
year=(010203)
epochs=(ep30)
project_major=(AMZ)
project_minor=(tl_bogota)
psize_train=(24)
psize_target=(24)
cell=(128)
experiment=(4_local)
ckp=(169921)
batchsize=24
SPLITS=(0)
REFERENCES=(MCD12Q1v6stable01to15_LCProp2_major)
optimizertype=(adam)
LR=(9.1346118342e-06)

for reference in ${REFERENCES[@]}; do
    for model in ${MODELS[@]}; do
        for split in ${SPLITS[@]}; do

            if [ "$model" = "bands" ]; then
                bands250m=(2)
                bands500m=(5)
            elif [ "$model" = "bands250m" ]; then
                 bands250m=(2)
                 bands500m=(1)
            elif [ "$model" = "bandswoblue" ]; then
                 bands250m=(2)
                 bands500m=(4)
            elif [ "$model" = "bandsaux" ]; then
                 bands250m=(5)
                 bands500m=(5)
            elif [ "$model" = "evi2" ]; then
                 bands250m=(1)
                 bands500m=(1)
            elif [ "$model" = "evi2" ]; then
                 bands250m=(1)
                 bands500m=(1)
            elif [ "$model" = "indices" ]; then
                 bands250m=(7)
                 bands500m=(5)
            else
                 bands250m=(10)
                 bands500m=(5)
            fi

            if [ "$reference" = "MCD12Q1v6raw_LCType1" ] || [ "$reference" = "MCD12Q1v6stable_LCType1" ] ; then
                 num_classes=(17)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp1" ]|| [ "$reference" = "MCD12Q1v6stable_LCProp1" ]; then
                 num_classes=(16)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp2" ] || [ "$reference" = "MCD12Q1v6stable_LCProp2" ]; then
                 num_classes=(11)
            elif [ "$reference" = "ESAraw" ] || [ "$reference" = "ESAstable" ]; then
                 num_classes=(37)
            elif [ "$reference" = "Copernicusraw" ] || [ "$reference" = "Copernicusnew_all2ofu" ] || [ "$reference" = "Copernicusnew_cebf2ofu" ] || [ "$reference" = "Copernicusnew_all2ofg" ] || [ "$reference" = "Copernicusraw_fraction" ]; then
                 num_classes=(22)
            elif [ "$reference" = "Copernicusnew_cf2others" ]; then
                 num_classes=(9)
            elif [ "$reference" = "merge_datasets2own" ] || [ "$reference" = "merge_datasets2HuHu" ] || [ "$reference" = "merge_datasets2Tsendbazaretal" ]; then
                 num_classes=(8)
            elif [ "$reference" = "MCD12Q1v6stable01to15_LCProp2_major" ] || [ "$reference" = "MCD12Q1v6stable01to15_LCProp2_major" ] || [ "$reference" = "MCD12Q1v6raw_LCProp2_major" ]; then
                 num_classes=(8)
            fi

            echo "Processing year: $year and model: $model and split: $split and reference: $reference"

            python modelzoo/seqencmodel.py \
                --modelfolder "E:/acocac/research/${project_minor}/models/$experiment/${model}/convgru${cell}_p${psize_target}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" \
                        --num_classes $num_classes \
                        --num_bands_250m $bands250m \
                        --num_bands_500m $bands500m \
                        --pix250m $psize_target \
                        --convrnn_filters=$cell \
                        --convcell gru \
                        --optimizertype $optimizertype \
                        --learning_rate $LR \
                        --convrnn_layers 1 \
                        --bidirectional TRUE

            python init_graph.py "E:/acocac/research/${project_minor}/models/$experiment/${model}/convgru${cell}_p${psize_target}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}/graph.meta"

            if [ "$ckp" = "last" ]; then
                python copy_network_weights.py "E:/acocac/research/${project_major}/models/$experiment/${epochs}/${model}/convgru${cell}_p${psize_train}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" "E:/acocac/research/${project_minor}/models/$experiment/${model}/convgru${cell}_p${psize_target}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" \
                --reset
            else
                python copy_network_weights.py "E:/acocac/research/${project_major}/models/$experiment/${epochs}/${model}/convgru${cell}_p${psize_train}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" "E:/acocac/research/${project_minor}/models/$experiment/${model}/convgru${cell}_p${psize_target}pxk0px_batch${batchsize}_${year}_${optimizertype}_${reference}_8d_l1_bidir_fold${split}" \
                    --sourcecheckpoint "model.ckpt-${ckp}" \
                    --reset
            fi
        done
    done
done