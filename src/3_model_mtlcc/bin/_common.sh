#!/bin/bash

SCALE_TIER="CUSTOM"
MASTER_MACHINE="large_model_v100" #complex_model_m_gpu standard_gpu
MODEL_NAME="mtlcc"
HPTUNING_CONFIG="trainer/hptuning_config.yaml"
BUCKET_NAME="thesis-2019"
PROJECT="tile_0_563"
PZISE_train=24
PZISE_eval=384
REGION="us-central1"
#export TF_FORCE_GPU_ALLOW_GROWTH=true

function get_date_time {
  echo "$(date +%Y%m%d%H%M%S)"
}