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

import argparse
import sys
from pathlib import Path

import tensorflow as tf

from efficientnet import eval_ckpt_main


def convert_tensorflow_model(
    model_name, model_ckpt, output_path, example_img="./assets/example_input.jpg"
):
    """ Loads and saves a TensorFlow model. """
    image_files = [example_img]
    eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
    with tf.get_default_graph().as_default():
        with tf.Session() as sess:
            images, _ = eval_ckpt_driver.build_dataset(
                image_files, [0] * len(image_files), False
            )
            inputs = {"input_images": tf.saved_model.utils.build_tensor_info(images)}
            probs = eval_ckpt_driver.build_model(images, is_training=False)
            outputs = {"outputs": tf.saved_model.utils.build_tensor_info(probs)}
            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.PREDICT_METHOD_NAME,
            )
            sess.run(tf.global_variables_initializer())
            eval_ckpt_driver.restore_model(sess, model_ckpt)
            builder = tf.saved_model.builder.SavedModelBuilder(output_path)
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
                },
                strip_default_attrs=True,
            )
            builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the EfficientNet checkpoint to SavedModel"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet-b0",
        help="efficientnet-b{N}, where N is an integer 0 <= N <= 7",
    )
    parser.add_argument(
        "--tf_checkpoint",
        type=str,
        default="./pretrained_tensorflow/efficientnet-b0/",
        help="checkpoint file path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./pretrained_keras/efficientnet-b0",
        help="output Keras model file name",
    )
    args = parser.parse_args()

    true_values = ("yes", "true", "t", "1", "y")
    convert_tensorflow_model(
        model_name=args.model_name,
        model_ckpt=args.tf_checkpoint,
        output_path=args.output_path,
    )
