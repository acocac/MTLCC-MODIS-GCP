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
"""Constants and common methods for preprocessing and training scripts."""

# file/folders patterns
MODEL_GRAPH_NAME = "graph.meta"
TRAINING_IDS_IDENTIFIER = "train"
TESTING_IDS_IDENTIFIER = "test"
EVAL_IDS_IDENTIFIER = "eval"

MODEL_CFG_FILENAME = "params.ini"
MODEL_CFG_FLAGS_SECTION = "flags"
MODEL_CFG_MODEL_SECTION = "model"
MODEL_CFG_MODEL_KEY = "model"

MODEL_CHECKPOINT_NAME = "model.ckpt"
TRAINING_SUMMARY_FOLDER_NAME = "train"
TESTING_SUMMARY_FOLDER_NAME = "test"
ADVANCED_SUMMARY_COLLECTION_NAME="advanced_summaries"

MASK_FOLDERNAME="mask"
GROUND_TRUTH_FOLDERNAME="ground_truth"
PREDICTION_FOLDERNAME="prediction"
LOSS_FOLDERNAME="loss"
CONFIDENCES_FOLDERNAME="confidences"
TRUE_PRED_FILENAME="truepred.npy"

graph_created_flag = False
