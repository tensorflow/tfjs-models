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
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model.load import load
from tensorflowjs import quantization
from tensorflowjs.converters import common, tf_saved_model_conversion_v2

from efficientnet import eval_ckpt_main

tf.compat.v1.disable_eager_execution()


# def _check_signature_in_model(saved_model, signature_name):
#     if signature_name not in saved_model.signatures:
#         raise ValueError(
#             "Signature '%s' does not exist. The following signatures "
#             "are available: %s" % (signature_name, saved_model.signatures.keys())
#         )


# def convert_tf_saved_model(
#     saved_model_dir,
#     output_dir,
#     signature_def="serving_default",
#     saved_model_tags="serve",
#     quantization_dtype=None,
#     skip_op_check=False,
#     strip_debug_ops=False,
# ):
#     """Freeze the SavedModel and check the model compatibility with Tensorflow.js.
#   Optimize and convert the model to Tensorflow.js format, when the model passes
#   the compatiblity check.
#   Args:
#     saved_model_dir: string The saved model directory.
#     : string The names of the output nodes, comma separated.
#     output_dir: string The name of the output directory. The directory
#       will consist of
#       - a file named 'model.json'
#       - possibly sharded binary weight files.
#     signature_def: string Tagset of the SignatureDef to load. Defaults to
#       'serving_default'.
#     saved_model_tags: tags of the GraphDef to load. Defaults to 'serve'.
#     quantization_dtype: An optional numpy dtype to quantize weights to for
#       compression. Only np.uint8 and np.uint16 are supported.
#     skip_op_check: Bool whether to skip the op check.
#     strip_debug_ops: Bool whether to strip debug ops.
#   """
#     if signature_def is None:
#         signature_def = "serving_default"

#     if not Path(output_dir).exists():
#         Path(output_dir).mkdir(parents=True)
#     output_graph = Path(output_dir) / common.ARTIFACT_MODEL_JSON_FILE_NAME

#     saved_model_tags = saved_model_tags.split(", ")
#     model = load(saved_model_dir, saved_model_tags)

#     tf_saved_model_conversion_v2._check_signature_in_model(model, signature_def)

#     concrete_func = model.signatures[signature_def]
#     frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

#     tf_saved_model_conversion_v2.optimize_graph(
#         frozen_func,
#         output_graph,
#         model.tensorflow_version,
#         quantization_dtype=quantization_dtype,
#         skip_op_check=skip_op_check,
#         strip_debug_ops=strip_debug_ops,
#     )


def convert_tensorflow_checkpoints(
    model_name, model_ckpt, saved_model_path, example_img="./assets/example_input.jpg"
):
    """ Loads and saves a TensorFlow model. """
    image_files = [example_img]
    eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            images, _ = eval_ckpt_driver.build_dataset(
                image_files, [0] * len(image_files), False
            )
            inputs = {
                "input_images": tf.compat.v1.saved_model.utils.build_tensor_info(images)
            }
            probs = eval_ckpt_driver.build_model(images, is_training=False)
            outputs = {
                "outputs": tf.compat.v1.saved_model.utils.build_tensor_info(probs)
            }
            signature = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs,
                outputs=outputs,
                method_name=tf.saved_model.PREDICT_METHOD_NAME,
            )
            sess.run(tf.compat.v1.global_variables_initializer())
            eval_ckpt_driver.restore_model(sess, model_ckpt)
            builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
                saved_model_path
            )
            builder.add_meta_graph_and_variables(
                sess,
                [tf.saved_model.SERVING],
                signature_def_map={
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
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
        "--saved_model_path",
        type=str,
        default="./pretrained_keras/efficientnet-b0",
        help="SavedModel target directory",
    )
    parser.add_argument(
        "--tfjs_model_path",
        type=str,
        default="./pretrained_keras/efficientnet-b0",
        help="TF.js JSON target directory",
    )
    args = parser.parse_args()

    if not Path(args.saved_model_path).exists():
        convert_tensorflow_checkpoints(
            model_name=args.model_name,
            model_ckpt=args.tf_checkpoint,
            saved_model_path=args.saved_model_path,
        )

    print("Converting the model to TF.js JSON...")
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        args.saved_model_path, args.tfjs_model_path
    )

    print("Converting and quantizing the model to TF.js JSON...")
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        args.saved_model_path,
        Path(args.tfjs_model_path).parent
        / "quantized"
        / Path(args.tfjs_model_path).stem,
        quantization_dtype=quantization.QUANTIZATION_BYTES_TO_DTYPES[2],
    )
