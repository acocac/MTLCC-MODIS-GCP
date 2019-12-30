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

# Convenience script for training model on AI Platform.
#
# Arguments:
#   MODEL_INPUTS_DIR: The directory containing the TFRecords from preprocessing.
#                     This should just be a timestamp.
. ./bin/_common.sh

NOW="$(get_date_time)"
INPUT_BUCKET="gs://${BUCKET_NAME}"
OUTPUT_BUCKET="gs://${BUCKET_NAME}"
TRAINING_JOB_NAME="${PROJECT}_train_${NOW}"
INPUT_PATH="${INPUT_BUCKET}/${PROJECT}/gz/${PZISE_train}/multiple"
MODEL_PATH="${OUTPUT_BUCKET}/${TRAINING_JOB_NAME}"
EPOCHS=1

TRAIN_YEAR='2001'
REFERENCE="MCD12Q1v6stable01to03_LCProp2"

CELL=(64)
LAYERS=(3)
LR=(0.001)
BS=(32)

gcloud ai-platform jobs submit training "${TRAINING_JOB_NAME}" \
  --module-name trainer.task \
  --staging-bucket "${OUTPUT_BUCKET}" \
  --package-path ./trainer \
  --region "${REGION}" \
  --runtime-version 1.15 \
  --scale-tier "${SCALE_TIER}" \
  --master-machine-type "${MASTER_MACHINE}" \
  -- \
  --modeldir "${MODEL_PATH}" \
  --datadir "${INPUT_PATH}" \
  --train_on "${TRAIN_YEAR}" \
  --epochs "${EPOCHS}" \
  --batchsize ${BS} \
  --reference ${REFERENCE} \
  --pix250m ${PZISE_train} \
  --convrnn_filters ${CELL} \
  --convrnn_layers ${LAYERS} \
  --learning_rate ${LR} \

echo "Upon completion, serve the model by running: bin/run.serve.cloud.sh ${NOW}"


#  --master-machine-type n1-standard-32 \
#  --master-accelerator count=1,type=nvidia-tesla-k80 \
#  --worker-count 2 \
#  --worker-machine-type n1-standard-4 \
#  --worker-accelerator count=1,type=nvidia-tesla-t4 \
#  --parameter-server-count 1 \
#  --parameter-server-machine-type n1-highmem-2 \