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

#MODEL_DIR=./sample/model
#INPUT_PATH=./sample/data/${PZISE_train}/multiple
INPUT_PATH="F:/acoca/research/gee/dataset/${PROJECT}/gz/${PZISE_train}/multiple"
MODEL_DIR="E:/acocac/research/${PROJECT}/models/2_gcloud"

EPOCHS=1
TRAIN_YEAR='2001'
#TRAIN_YEAR=(2015)
REFERENCE="MCD12Q1v6stable01to03_LCProp2"

CELL=(128)
LAYERS=(1)
LR=(0.01)
BS=(32)

mkdir -p "${MODEL_DIR}/_logs"
logfname="${MODEL_DIR}/_logs/log.log"

gcloud ai-platform local train \
  --module-name trainer.task \
  --package-path trainer \
  -- \
  --modeldir "${MODEL_DIR}" \
  --datadir "${INPUT_PATH}" \
  --train_on "${TRAIN_YEAR}" \
  --reference ${REFERENCE} \
  --epochs "${EPOCHS}" \
  --pix250m "${PZISE_train}" \
  --convrnn_filters ${CELL} \
  --convrnn_layers ${LAYERS} \
  --learning_rate "${LR}" \
  --step 'training' \
  --batchsize ${BS} > $logfname 2>&1

echo "Upon completion, serve the model by running: bin/run.serve.local.sh"

#  --limit_batches 1
