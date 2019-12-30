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

#MODEL_DIR=./sample/model
#INPUT_PATH=./sample/data/gz/${PZISE}/multiple
#INPUT_PATH="F:/acoca/research/gee/dataset/AMZ/MOD09_250m500m/gz/${PZISE_eval}/multiple"
INPUT_PATH="F:/acoca/research/gee/dataset/${PROJECT}/gz/${PZISE_eval}/multiple"
MODEL_DIR="E:/acocac/research/${PROJECT}/models/2_gcloud/t2003_1to18"
STORE_DIR="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/t2003_1to18"
REFERENCE="MCD12Q1v6stable01to18_LCProp2"

EPOCHS=20
YEAR=(2001)

CELL=(64)
LAYERS=(1)
LR=(0.01)
BS=(1)

mkdir -p "E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs"
logfname="E:/acocac/research/${PROJECT}/eval/pred/2_gcloud/_logs/log.log"

gcloud ai-platform local train \
  --module-name trainer.eval \
  --package-path trainer \
  -- \
  --modeldir "${MODEL_DIR}" \
  --datadir "${INPUT_PATH}" \
  --storedir "${STORE_DIR}/${YEAR}" \
  --dataset "${YEAR}" \
  --reference ${REFERENCE} \
  --epochs "${EPOCHS}" \
  --pix250m "${PZISE_eval}" \
  --convrnn_filters ${CELL} \
  --convrnn_layers ${LAYERS} \
  --learning_rate "${LR}" \
  --writetiles \
  --writeconfidences \
  --step 'evaluation' \
  --batchsize ${BS} > $logfname 2>&1

echo "Upon completion, serve the model by running: bin/run.serve.local.sh"
