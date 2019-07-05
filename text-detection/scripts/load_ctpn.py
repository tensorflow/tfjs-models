# coding=utf-8
import argparse
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

import cv2
from nets import model_train as model

sys.path.append(os.getcwd())


def convert_to_saved_model(checkpoint_path, output_path):
    os.makedirs(output_path)

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name="input_image"
        )
        input_im_info = tf.placeholder(
            tf.float32, shape=[None, 3], name="input_im_info"
        )

        inputs = {
            "input_image": tf.saved_model.utils.build_tensor_info(input_image),
            "input_im_info": tf.saved_model.utils.build_tensor_info(input_im_info),
        }

        global_step = tf.get_variable(
            "global_step", [], initializer=tf.constant_initializer(0), trainable=False
        )

        bbox_pred, _, cls_prob = model.model(input_image)

        outputs = {
            "bbox_pred": tf.saved_model.utils.build_tensor_info(bbox_pred),
            "cls_prob": tf.saved_model.utils.build_tensor_info(cls_prob),
        }

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(
                checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path)
            )
            saver.restore(sess, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert the CTPN checkpoint to SavedModel"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="pretrained_tensorflow/efficientnet-b0/",
        help="The checkpoint path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="pretrained_keras/efficientnet-b0",
        help="The SavedModel path",
    )
    args = parser.parse_args()
    convert_to_saved_model(checkpoint_path=args.tf_checkpoint, output_path=output_path)
