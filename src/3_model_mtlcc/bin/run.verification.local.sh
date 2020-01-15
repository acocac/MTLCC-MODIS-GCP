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

YEARS=(2001)
input=(bands)
REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
PROJECTS=(AMZ)

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
              --experiment=$input \
              --reference=$reference \
              --step='verification' \
              --allow_growth TRUE > $logfname 2>&1
        done
    done
done