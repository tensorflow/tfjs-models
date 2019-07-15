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
# ==============================================================================
# coding=utf-8
import argparse
import os
import time

import tensorflow as tf

from psenet import model


def convert_to_saved_model(checkpoint_path, output_path):
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="input_images"
        )

        inputs = {"input_images": tf.saved_model.utils.build_tensor_info(input_images)}

        seg_maps_pred = model.model(input_images, outputs=3, is_training=False)
        outputs = {"seg_maps": tf.saved_model.utils.build_tensor_info(seg_maps_pred)}

        global_step = tf.get_variable(
            "global_step", [], initializer=tf.constant_initializer(0), trainable=False
        )
        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
        )
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        builder = tf.saved_model.builder.SavedModelBuilder(args.output_path)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            model_path = os.path.join(checkpoint_path, "model.ckpt")
            saver.restore(sess, model_path)
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
        description="Convert the PSENet checkpoint to SavedModel"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="dist/weights",
        help="The checkpoint path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="dist/saved_model",
        help="The SavedModel path",
    )

    args = parser.parse_args()
    convert_to_saved_model(
        checkpoint_path=args.checkpoint_path, output_path=args.output_path
    )
