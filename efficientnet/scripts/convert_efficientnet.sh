#!/usr/bin/env bash
# =============================================================================
# Copyright 2019 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Make sure that we run Python 3.6, not 3.7
PYTHON_VERSION=$(python -V 2>&1 | sed 's/.* \([0-9]\).\([0-9]\).*/\1\2/')
if [ "$PYTHON_VERSION" -gt "36" ]; then
  echo "This script requires python 3.6 or older."
  exit 1
fi

if [ -z "$1" ]; then
  echo "No target directory supplied, aborting."
  exit 1
fi

SOURCE_CODE_URL="https://raw.githubusercontent.com/tensorflow/tpu/master/models/official/"
SOURCE_CODE_FILES=(
  "efficientnet_builder.py"
  "efficientnet_model.py"
  "eval_ckpt_main.py"
  "utils.py"
  "preprocessing.py"
)
SOURCE_CODE_DIR="src"

CHECKPOINTS_DIR="pretrained_tensorflow"
CHECKPOINT_PREFIX="efficientnet-"
CHECKPOINTS_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/"
CHECKPOINTS_EXT=".tar.gz"

MODEL_NAME="efficientnet"
URL_PREFIX="https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-"
MODELS=(
  "b0"
  "b3"
  "b5"
)
FILE_SUFFIX="_imagenet_1000.h5"

MODEL_DIR="dist"
# MODEL_DIR=$(mktemp -d)
SCRIPT_DIR=$(pwd)

# trap 'rm -rf -- "$MODEL_DIR"' INT TERM HUP EXIT

printf -- '=%.0s' {1..80}
echo ""
echo "Setting up the installation environment..."
printf -- '=%.0s' {1..80}
echo ""

set -e

cd $MODEL_DIR &&
  echo 1
# virtualenv --no-site-packages venv && \
# source venv/bin/activate && \
# pip install tensorflowjs numpy tensorflow
printf -- '=%.0s' {1..80}
echo ""
echo "Downloading the checkpoints..."
printf -- '=%.0s' {1..80}
echo ""

mkdir -p $CHECKPOINTS_DIR
for MODEL_VERSION in "${MODELS[@]}"; do
  aria2c -x 16 -k 1M -o $MODEL_VERSION$CHECKPOINTS_EXT $CHECKPOINTS_URL$CHECKPOINT_PREFIX$MODEL_VERSION$CHECKPOINTS_EXT
  tar xvf $MODEL_VERSION$CHECKPOINTS_EXT
  rm $MODEL_VERSION$CHECKPOINTS_EXT
done

printf -- '=%.0s' {1..80}
echo ""
echo "Converting the checkpoints to Keras..."
printf -- '=%.0s' {1..80}
echo ""

mkdir -p $SOURCE_CODE_DIR
touch $SOURCE_CODE_DIR/__init__.py
for SOURCE_CODE_FILE in "${SOURCE_CODE_FILES[@]}"; do
  aria2c -x 16 -k 1M -o $SOURCE_CODE_DIR/$SOURCE_CODE_FILE $SOURCE_CODE_URL/$SOURCE_CODE_FILE
done

mkdir -p $SCRIPT_DIR/$1/$MODEL_NAME/quantized
# for MODEL_VERSION in "${MODELS[@]}"
# do
#   printf -- '=%.0s' {1..80}; echo ""

#   printf -- '-%.0s' {1..80}; echo ""
#   echo "Downloading and extracting the model from $URL_PREFIX$MODEL_VERSION$FILE_SUFFIX..."
#   printf -- '-%.0s' {1..80}; echo ""

#   aria2c -x 16 -k 1M -o $MODEL_VERSION$FILE_SUFFIX $URL_PREFIX$MODEL_VERSION$FILE_SUFFIX

#   printf -- '-%.0s' {1..80}; echo ""
#   echo "Converting the model to JSON..."
#   printf -- '-%.0s' {1..80}; echo ""

#   tensorflowjs_converter \
#     --input_format keras \
#     --output_format tfjs_graph_model \
#     $MODEL_DIR/$MODEL_VERSION$FILE_SUFFIX \
#     $1/$MODEL_NAME/$MODEL_VERSION

#   printf -- '-%.0s' {1..80}; echo ""
#   echo "Converting the model to quantized JSON..."
#   printf -- '-%.0s' {1..80}; echo ""

#   tensorflowjs_converter \
#     --quantization_bytes 2 \
#     --input_format keras \
#     --output_format tfjs_graph_model \
#     $MODEL_DIR/$MODEL_VERSION$FILE_SUFFIX \
#     $1/$MODEL_NAME/quantized/$MODEL_VERSION

#   printf -- '=%.0s' {1..80}; echo ""
# done

# echo "Success!"
tensorflowjs_converter \
  --quantization_bytes 2 \
  --input_format keras \
  --output_format tfjs_graph_model \
  efficientnet/dist/efficientnet-b0.h5 \
  testdir
