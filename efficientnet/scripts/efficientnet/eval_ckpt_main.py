# Copyright 2019 The TensorFlow Authors, Google Inc. All Rights Reserved.
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
"""Eval checkpoint driver.

This is an example evaluation script for users to understand the EfficientNet
model checkpoints on CPU. To serve EfficientNet, please consider to export a
`SavedModel` from checkpoints and use tf-serving to serve.
"""

from __future__ import absolute_import, division, print_function

import json
import sys

import numpy as np
import tensorflow as tf
from absl import app, flags

from . import efficientnet_builder, preprocessing

flags.DEFINE_string("model_name", "efficientnet-b0", "Model name to eval.")
flags.DEFINE_string("runmode", "examples", "Running mode: examples or imagenet")
flags.DEFINE_string(
    "imagenet_eval_glob",
    None,
    "Imagenet eval image glob, " "such as /imagenet/ILSVRC2012*.JPEG",
)
flags.DEFINE_string(
    "imagenet_eval_label",
    None,
    "Imagenet eval label file path, "
    "such as /imagenet/ILSVRC2012_validation_ground_truth.txt",
)
flags.DEFINE_string("ckpt_dir", "/tmp/ckpt/", "Checkpoint folders")
flags.DEFINE_boolean("enable_ema", True, "Enable exponential moving average.")
flags.DEFINE_string("export_ckpt", None, "Exported ckpt for eval graph.")
flags.DEFINE_string(
    "example_img", "/tmp/panda.jpg", "Filepath for a single example image."
)
flags.DEFINE_string(
    "labels_map_file", "/tmp/labels_map.txt", "Labels map from label id to its meaning."
)
flags.DEFINE_integer(
    "num_images", 5000, "Number of images to eval. Use -1 to eval all images."
)
FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class EvalCkptDriver(object):
    """A driver for running eval inference.

  Attributes:
    model_name: str. Model name to eval.
    batch_size: int. Eval batch size.
    num_classes: int. Number of classes, default to 1000 for ImageNet.
    image_size: int. Input image size, determined by model name.
  """

    def __init__(self, model_name="efficientnet-b0", batch_size=1):
        """Initialize internal variables."""
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = 1000
        # Model Scaling parameters
        _, _, self.image_size, _ = efficientnet_builder.efficientnet_params(model_name)

    def restore_model(self, sess, ckpt_dir, enable_ema=True, export_ckpt=None):
        """Restore variables from checkpoint dir."""
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if enable_ema:
            ema = tf.train.ExponentialMovingAverage(decay=0.0)
            ema_vars = tf.trainable_variables() + tf.get_collection("moving_vars")
            for v in tf.global_variables():
                if "moving_mean" in v.name or "moving_variance" in v.name:
                    ema_vars.append(v)
            ema_vars = list(set(ema_vars))
            var_dict = ema.variables_to_restore(ema_vars)
            ema_assign_op = ema.apply(ema_vars)
        else:
            var_dict = None
            ema_assign_op = None

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_dict, max_to_keep=1)
        saver.restore(sess, checkpoint)

        if export_ckpt:
            if ema_assign_op is not None:
                sess.run(ema_assign_op)
            saver = tf.train.Saver(max_to_keep=1)
            saver.save(sess, export_ckpt)

    def build_model(self, features, is_training):
        """Build model with input features."""
        features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
        features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
        logits, _ = efficientnet_builder.build_model(
            features, self.model_name, is_training
        )
        probs = tf.nn.softmax(logits)
        probs = tf.squeeze(probs)
        return probs

    def build_dataset(self, filenames, labels, is_training):
        """Build input dataset."""
        filenames = tf.constant(filenames)
        labels = tf.constant(labels)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = preprocessing.preprocess_image(
                image_string, is_training, image_size=self.image_size
            )
            image = tf.cast(image_decoded, tf.float32)
            return image, label

        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        return images, labels

    def run_inference(self, ckpt_dir, image_files, labels):
        """Build and run inference on the target images and labels."""
        with tf.Graph().as_default(), tf.Session() as sess:
            images, labels = self.build_dataset(image_files, labels, False)
            probs = self.build_model(images, is_training=False)

            self.restore_model(sess, ckpt_dir, FLAGS.enable_ema, FLAGS.export_ckpt)

            prediction_idx = []
            prediction_prob = []
            for _ in range(len(image_files) // self.batch_size):
                out_probs = sess.run(probs)
                idx = np.argsort(out_probs)[::-1]
                prediction_idx.append(idx[:5])
                prediction_prob.append([out_probs[pid] for pid in idx[:5]])

            # Return the top 5 predictions (idx and prob) for each image.
            return prediction_idx, prediction_prob


def eval_example_images(model_name, ckpt_dir, image_files, labels_map_file):
    """Eval a list of example images.

  Args:
    model_name: str. The name of model to eval.
    ckpt_dir: str. Checkpoint directory path.
    image_files: List[str]. A list of image file paths.
    labels_map_file: str. The labels map file path.

  Returns:
    A tuple (pred_idx, and pred_prob), where pred_idx is the top 5 prediction
    index and pred_prob is the top 5 prediction probability.
  """
    eval_ckpt_driver = EvalCkptDriver(model_name)
    classes = json.loads(tf.gfile.Open(labels_map_file).read())
    pred_idx, pred_prob = eval_ckpt_driver.run_inference(
        ckpt_dir, image_files, [0] * len(image_files)
    )
    for i in range(len(image_files)):
        print("predicted class for image {}: ".format(image_files[i]))
        for j, idx in enumerate(pred_idx[i]):
            print(
                "  -> top_{} ({:4.2f}%): {}  ".format(
                    j, pred_prob[i][j] * 100, classes[str(idx)]
                )
            )
    return pred_idx, pred_prob


def eval_imagenet(
    model_name, ckpt_dir, imagenet_eval_glob, imagenet_eval_label, num_images
):
    """Eval ImageNet images and report top1/top5 accuracy.

  Args:
    model_name: str. The name of model to eval.
    ckpt_dir: str. Checkpoint directory path.
    imagenet_eval_glob: str. File path glob for all eval images.
    imagenet_eval_label: str. File path for eval label.
    num_images: int. Number of images to eval: -1 means eval the whole dataset.

  Returns:
    A tuple (top1, top5) for top1 and top5 accuracy.
  """
    eval_ckpt_driver = EvalCkptDriver(model_name)
    imagenet_val_labels = [int(i) for i in tf.gfile.GFile(imagenet_eval_label)]
    imagenet_filenames = sorted(tf.gfile.Glob(imagenet_eval_glob))
    if num_images < 0:
        num_images = len(imagenet_filenames)
    image_files = imagenet_filenames[:num_images]
    labels = imagenet_val_labels[:num_images]

    pred_idx, _ = eval_ckpt_driver.run_inference(ckpt_dir, image_files, labels)
    top1_cnt, top5_cnt = 0.0, 0.0
    for i, label in enumerate(labels):
        top1_cnt += label in pred_idx[i][:1]
        top5_cnt += label in pred_idx[i][:5]
        if i % 100 == 0:
            print(
                "Step {}: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%".format(
                    i, 100 * top1_cnt / (i + 1), 100 * top5_cnt / (i + 1)
                )
            )
            sys.stdout.flush()
    top1, top5 = 100 * top1_cnt / num_images, 100 * top5_cnt / num_images
    print("Final: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%".format(top1, top5))
    return top1, top5


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.ERROR)
    if FLAGS.runmode == "examples":
        # Run inference for an example image.
        eval_example_images(
            FLAGS.model_name, FLAGS.ckpt_dir, [FLAGS.example_img], FLAGS.labels_map_file
        )
    elif FLAGS.runmode == "imagenet":
        # Run inference for imagenet.
        eval_imagenet(
            FLAGS.model_name,
            FLAGS.ckpt_dir,
            FLAGS.imagenet_eval_glob,
            FLAGS.imagenet_eval_label,
            FLAGS.num_images,
        )
    else:
        print("must specify runmode: examples or imagenet")


if __name__ == "__main__":
    app.run(main)
