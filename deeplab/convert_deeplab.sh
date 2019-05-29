#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "No argument supplied, aborting."
    exit 1
fi

URL_PREFIX="http://download.tensorflow.org/models"
MODELS=(
  "deeplabv3_mnv2_dm05_pascal_trainaug_2018_10_01"
  "deeplabv3_mnv2_cityscapes_train_2018_02_05"
  "deeplabv3_mnv2_ade20k_train_2018_12_03"
)
URL_SUFFIX=".tar.gz"

MODEL_DIR=$(mktemp -d)
SCRIPT_DIR=$(pwd)

DIST_DIR='./dist'

trap 'rm -rf -- "$MODEL_DIR"' INT TERM HUP EXIT

mkdir -p $SCRIPT_DIR/$1/deeplab && \
cd $MODEL_DIR && \
pyenv local 3.6.8 && \
virtualenv --no-site-packages venv && \
source venv/bin/activate && \
pip install tensorflowjs==0.8.6 && \
echo "Downloading models to $MODEL_DIR..."

for MODEL_NAME in "${MODELS[@]}"
do
  wget -P $MODEL_DIR $URL_PREFIX/$MODEL_NAME$URL_SUFFIX && \
  mkdir -p $MODEL_DIR/$MODEL_NAME && \
  tar -xvzf $MODEL_DIR/$MODEL_NAME$URL_SUFFIX -C $MODEL_DIR/$MODEL_NAME --strip-components 1 && \

  OUTPUT_DIR=""

  if [[ $MODEL_NAME == *"pascal"* ]]
  then
    OUTPUT_DIR="pascal"
  elif [[ $MODEL_NAME == *"cityscapes"* ]]
  then
    OUTPUT_DIR="cityscapes"
  elif [[ $MODEL_NAME == *"ade20k"* ]]
  then
    OUTPUT_DIR="ade20k"
  else
    echo "The model is not supported, aborting."
    exit 1
  fi

  tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_json=true \
    --saved_model_tags=serve \
    --output_node_names='SemanticPredictions' \
    $MODEL_DIR/$MODEL_NAME/frozen_inference_graph.pb \
    $SCRIPT_DIR/$1/deeplab/$OUTPUT_DIR
done
