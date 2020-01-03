#!/bin/bash

. ./bin/_common.sh

NOW="$(get_date_time)"
INPUT_BUCKET="gs://${BUCKET_NAME}"
OUTPUT_BUCKET="gs://${BUCKET_NAME}"
TRAINING_JOB_NAME="${PROJECT}_hpt_train_${NOW}"
INPUT_PATH="${INPUT_BUCKET}/${PROJECT}/${PZISE_train}/multiple"
MODEL_PATH="${OUTPUT_BUCKET}/${TRAINING_JOB_NAME}"
EPOCHS=1
TRAIN_YEAR='2001'

NUM_GPUS_IN_MASTER=1

gcloud ai-platform jobs submit training "${TRAINING_JOB_NAME}" \
  --module-name trainer.task \
  --staging-bucket "${OUTPUT_BUCKET}" \
  --package-path trainer \
  --region "${REGION}" \
  --runtime-version 1.14 \
  --config ${HPTUNING_CONFIG} \
  --scale-tier "${SCALE_TIER}" \
  --master-machine-type n1-highmem-8 \
  --master-accelerator count=$NUM_GPUS_IN_MASTER,type=nvidia-tesla-p100 \
  -- \
  --modeldir "${MODEL_PATH}" \
  --datadir "${INPUT_PATH}" \
  --train_on "${TRAIN_YEAR}" \
  --epochs "${EPOCHS}" \
  --pix250m ${PZISE_train} \
  --epochs "${EPOCHS}" \

#echo "Upon completion, serve the model by running: bin/run.serve.sh ${NOW}"