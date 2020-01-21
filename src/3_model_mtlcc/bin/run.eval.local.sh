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
# notes acc: eval.py working

. ./bin/_common.sh

REFERENCES=(MCD12Q1v6stable01to03_LCProp2_major)
YEARS=(2001)
FOLDS=(0)
BS=1

CELL=(64)
LAYERS=(1)
LR=(2.0367720993e-05)
optimizertype=(adam)
experiment=(bands)

TRIALID=(AMZ_hpt_train_20200118154937)
BESTTRIAL=(1)
experiment=(bands)

for reference in ${REFERENCES[@]}; do

    for fold in ${FOLDS[@]}; do

        INPUT_PATH="F:/acoca/research/gee/dataset/${PROJECT}/gz/${PZISE_eval}/multiple"
        MODEL_DIR="E:/acocac/research/${PROJECT}/models/2_gcloud/$TRIALID/$BESTTRIAL"
        STORE_DIR="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/$TRIALID/$BESTTRIAL"

        for year in ${YEARS[@]}; do
            echo "Processing project: $PROJECT and year: $year and reference: $reference and trial $TRIALID folder $BESTTRIAL"

            mkdir -p "E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs/$TRIALID/"
            logfname="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs/$TRIALID/$BESTTRIAL_$reference_$year.log"

            gcloud ai-platform local train \
              --module-name trainer.eval \
              --package-path trainer \
              -- \
              --modeldir "${MODEL_DIR}" \
              --datadir "${INPUT_PATH}" \
              --storedir "${STORE_DIR}/fold${fold}/${year}" \
              --dataset "${year}" \
              --reference ${reference} \
              --pix250m "${PZISE_eval}" \
              --convrnn_filters ${CELL} \
              --convrnn_layers ${LAYERS} \
              --learning_rate ${LR} \
              --experiment ${experiment} \
              --writetiles \
              --step 'evaluation' \
              --optimizertype ${optimizertype} \
              --batchsize ${BS} > $logfname 2>&1
        done
    done
done
#          --writeconfidence