#!/bin/bash

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Convenience script for training model locally.
#
# Arguments:
#   MODEL_INPUTS_DIR: The directory containing the TFRecords from preprocessing.
#                     This should just be a timestamp.

. ./bin/_common.sh

#YEARS=(2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017)
YEARS=(2001)
experiment=(2_gcloud)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2)
PROJECTS=(tile_0_563)

for reference in ${REFERENCES[@]}; do
    for project in ${PROJECTS[@]}; do
        echo $reference
        mkdir -p "E:/acocac/research/${project}/eval/verification"

        if [ "$reference" = "mapbiomas_fraction" ]; then
            foldername=("mapbiomas")
        elif [ "$model" = "bands250m" ]; then
            foldername=("watermask")
        else
            foldername=$reference
        fi

        mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername"
        mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername/_logs"

        for year in ${YEARS[@]}; do
            echo "Processing project: $project and year: $year and reference: $reference"

            mkdir -p "E:/acocac/research/${project}/eval/verification/$foldername/$year"

            logfname="E:/acocac/research/${project}/eval/verification/$foldername/_logs/$year.log"
            gcloud ai-platform local train \
              --module-name trainer.verification \
              --package-path trainer \
              -- \
              --datadir="F:/acoca/research/gee/dataset/${project}/gz/${PZISE_eval}/multiple" \
              --storedir="E:/acocac/research/${project}/eval/verification/$foldername/$year" \
              --writetiles \
              --batchsize=1 \
              --dataset=$year \
              --experiment='bands' \
              --reference=$reference \
              --step='verification' \
              --allow_growth TRUE > $logfname 2>&1
        done
    done
done
#
##MODEL_DIR=./sample/model
##INPUT_PATH=./sample/data/gz/${PZISE}/multiple
##INPUT_PATH="F:/acoca/research/gee/dataset/AMZ/MOD09_250m500m/gz/${PZISE_eval}/multiple"
#INPUT_PATH="F:/acoca/research/gee/dataset/${PROJECT}/gz/${PZISE_eval}/multiple"
#MODEL_DIR="E:/acocac/research/${PROJECT}/models/2_gcloud/t2003_1to18"
#STORE_DIR="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/t2003_1to18"
#REFERENCE="MCD12Q1v6stable01to18_LCProp2"
#
#EPOCHS=20
#YEAR=(2001)
#
#CELL=(64)
#LAYERS=(1)
#LR=(0.01)
#BS=(1)
#
#mkdir -p "E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs"
#logfname="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs/log.log"
#
#gcloud ai-platform local train \
#  --module-name trainer.eval \
#  --package-path trainer \
#  -- \
#  --modeldir "${MODEL_DIR}" \
#  --datadir "${INPUT_PATH}" \
#  --storedir "${STORE_DIR}/${YEAR}" \
#  --dataset "${YEAR}" \
#  --reference ${REFERENCE} \
#  --epochs "${EPOCHS}" \
#  --pix250m "${PZISE_eval}" \
#  --convrnn_filters ${CELL} \
#  --convrnn_layers ${LAYERS} \
#  --learning_rate "${LR}" \
#  --writetiles \
#  --writeconfidences \
#  --batchsize ${BS} > $logfname 2>&1
#
#echo "Upon completion, serve the model by running: bin/run.serve.local.sh"