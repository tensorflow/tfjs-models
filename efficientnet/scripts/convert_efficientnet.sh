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

if [ -z "$1" ]
  then
    echo "No target directory supplied, aborting."
    exit 1
fi


MODEL_NAME="efficientnet"
URL_PREFIX="https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-"
MODELS=(
  "b0"
  "b3"
)
FILE_SUFFIX="_imagenet_1000.h5"

MODEL_DIR=$(mktemp -d)
SCRIPT_DIR=$(pwd)

DIST_DIR='./dist'

trap 'rm -rf -- "$MODEL_DIR"' INT TERM HUP EXIT

printf -- '=%.0s' {1..80}; echo ""
echo "Setting up the installation environment..."
printf -- '=%.0s' {1..80}; echo ""

set -e
mkdir -p $SCRIPT_DIR/$1/$MODEL_NAME/quantized && \
cd $MODEL_DIR && \
echo 1
# virtualenv --no-site-packages venv && \
# source venv/bin/activate && \
# pip install tensorflowjs

for MODEL_VERSION in "${MODELS[@]}"
do
  printf -- '=%.0s' {1..80}; echo ""

  printf -- '-%.0s' {1..80}; echo ""
  echo "Downloading and extracting the model from $URL_PREFIX$MODEL_VERSION$FILE_SUFFIX..."
  printf -- '-%.0s' {1..80}; echo ""

  aria2c -x 16 -k 1M -o $MODEL_VERSION$FILE_SUFFIX $URL_PREFIX$MODEL_VERSION$FILE_SUFFIX

  printf -- '-%.0s' {1..80}; echo ""
  echo "Converting the model to JSON..."
  printf -- '-%.0s' {1..80}; echo ""

  tensorflowjs_converter \
    --input_format=keras \
    $MODEL_DIR/$MODEL_VERSION$FILE_SUFFIX \
    $SCRIPT_DIR/$1/$MODEL_NAME/$MODEL_VERSION

  printf -- '-%.0s' {1..80}; echo ""
  echo "Converting the model to quantized JSON..."
  printf -- '-%.0s' {1..80}; echo ""

  tensorflowjs_converter \
    --quantization_bytes 2 \
    --input_format=keras \
    $MODEL_DIR/$MODEL_VERSION$FILE_SUFFIX \
    $SCRIPT_DIR/$1/$MODEL_NAME/quantized/$MODEL_VERSION

  printf -- '=%.0s' {1..80}; echo ""
done

echo "Success!"
