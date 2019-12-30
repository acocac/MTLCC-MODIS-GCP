#!/bin/bash

. ./bin/_common.sh

MODELS=(bands)
#MODELS=(all)
year=(15)
epochs=(ep5)
project=(AMZ)
psize_train=(24)
psize_eval=(384)
experiment=(0_estimator)
ckp=(7080)
batchsize=32
SPLITS=(0)
REFERENCES=(MCD12Q1v6raw_LCType1)

cell=(32)
LAYERS=(1)
LR=(0.01)

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

            if [ "$reference" = "MCD12Q1v6raw_LCType1" ]; then
                 num_classes=(17)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp1" ]; then
                 num_classes=(16)
            elif [ "$reference" = "MCD12Q1v6raw_LCProp2" ]; then
                 num_classes=(11)
            elif [ "$reference" = "ESAraw" ]; then
                 num_classes=(37)
            elif [ "$reference" = "Copernicusraw" ]; then
                 num_classes=(22)
            elif [ "$reference" = "Copernicusnew_all" ]; then
                 num_classes=(22)
            elif [ "$reference" = "Copernicusnew_cebf" ]; then
                 num_classes=(22)
            fi

            echo "Processing year: $year and model: $model and split: $split and reference: $reference"

            python experiments/seqencmodel.py \
                --modelfolder "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/eval/models/$experiment/${epochs}/convgru${cell}_p${psize_eval}pxk0px_batch${batchsize}_${year}_${reference}_8d_l1_bidir_${model}_fold${split}_ckp${ckp}" \
                        --num_classes $num_classes \
                        --num_bands_250m $bands250m \
                        --num_bands_500m $bands500m \
                        --pix250m $psize_eval \
                        --convrnn_filters=$cell \
                        --convcell gru \
                        --convrnn_layers "${LAYERS}" \
                        --learning_rate "${LR}" \
                        --bidirectional TRUE

            python experiments/init_graph.py "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/eval/models/$experiment/${epochs}/convgru${cell}_p${psize_eval}pxk0px_batch${batchsize}_${year}_${reference}_8d_l1_bidir_${model}_fold${split}_ckp${ckp}/graph.meta"

            if [ "$ckp" = "last" ]; then
                python experiments/copy_network_weights.py "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/${psize_train}/model" "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/eval/models/$experiment/${epochs}/convgru${cell}_p${psize_eval}pxk0px_batch${batchsize}_${year}_${reference}_8d_l1_bidir_${model}_fold${split}_ckp${ckp}"
            else
                python experiments/copy_network_weights.py "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/${psize_train}/model" "E:/acocac/research/scripts/thesis_cloud/3c_model_mtlcc_estimator/${project}/eval/models/$experiment/${epochs}/convgru${cell}_p${psize_eval}pxk0px_batch${batchsize}_${year}_${reference}_8d_l1_bidir_${model}_fold${split}_ckp${ckp}" \
                    --sourcecheckpoint "model.ckpt-${ckp}"
            fi
        done
    done
done