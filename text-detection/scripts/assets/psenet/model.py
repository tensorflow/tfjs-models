# Copyright 2019 Google Inc, Liu Heng. All Rights Reserved.
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
# -*- coding:utf-8 -*-
import tensorflow as tf

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer("text_scale", 512, "")

from .resnet import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs, rate):
    return tf.image.resize_bilinear(
        inputs, size=[tf.shape(inputs)[1] * rate, tf.shape(inputs)[2] * rate]
    )


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    """
    image normalization
    :param images:
    :param means:
    :return:
    """
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("len(means) must match the number of channels")
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def build_feature_pyramid(C, weight_decay):

    """
    reference: https://github.com/CharlesShang/FastMaskRCNN
    build P2, P3, P4, P5
    :return: multi-scale feature map
    """

    feature_pyramid = {}
    with tf.variable_scope("build_feature_pyramid"):
        with slim.arg_scope(
            [slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)
        ):
            feature_pyramid["P5"] = slim.conv2d(
                C["C5"], num_outputs=256, kernel_size=[1, 1], stride=1, scope="build_P5"
            )

            # feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
            #                                         kernel_size=[2, 2], stride=2, scope='build_P6')
            # P6 is down sample of P5

            for layer in range(4, 1, -1):
                p, c = feature_pyramid["P" + str(layer + 1)], C["C" + str(layer)]
                up_sample_shape = tf.shape(c)
                up_sample = tf.image.resize_nearest_neighbor(
                    p,
                    [up_sample_shape[1], up_sample_shape[2]],
                    name="build_P%d/up_sample_nearest_neighbor" % layer,
                )

                c = slim.conv2d(
                    c,
                    num_outputs=256,
                    kernel_size=[1, 1],
                    stride=1,
                    scope="build_P%d/reduce_dimension" % layer,
                )
                p = up_sample + c
                p = slim.conv2d(
                    p,
                    256,
                    kernel_size=[3, 3],
                    stride=1,
                    padding="SAME",
                    scope="build_P%d/avoid_aliasing" % layer,
                )
                feature_pyramid["P" + str(layer)] = p
    return feature_pyramid


def model(images, outputs=6, weight_decay=1e-5, is_training=True):
    """
    define the model, we use slim's implemention of resnet
    """
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(
            images, is_training=is_training, scope="resnet_v1_50"
        )

    # no non-linearities in FPN article
    feature_pyramid = build_feature_pyramid(end_points, weight_decay=weight_decay)
    # unpool sample P
    P_concat = []
    for i in range(3, 0, -1):
        P_concat.append(unpool(feature_pyramid["P" + str(i + 2)], 2 ** i))
    P_concat.append(feature_pyramid["P2"])
    # F = C(P2,P3,P4,P5)
    F = tf.concat(P_concat, axis=-1)

    # reduce to 256 channels
    with tf.variable_scope("feature_results"):
        batch_norm_params = {
            "decay": 0.997,
            "epsilon": 1e-5,
            "scale": True,
            "is_training": is_training,
        }
        with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            weights_regularizer=slim.l2_regularizer(weight_decay),
        ):
            F = slim.conv2d(F, 256, 3)
        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            activation_fn=None,
        ):
            S = slim.conv2d(F, outputs, 1)

    seg_S_pred = tf.nn.sigmoid(S)

    return seg_S_pred


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    """
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = (
        tf.reduce_sum(y_true_cls * training_mask)
        + tf.reduce_sum(y_pred_cls * training_mask)
        + eps
    )
    dice = 2 * intersection / union
    loss = 1.0 - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice, loss


def loss(y_true_cls, y_pred_cls, training_mask):
    g1, g2, g3, g4, g5, g6 = tf.split(value=y_true_cls, num_or_size_splits=6, axis=3)
    s1, s2, s3, s4, s5, s6 = tf.split(value=y_pred_cls, num_or_size_splits=6, axis=3)
    Gn = [g1, g2, g3, g4, g5, g6]
    Sn = [s1, s2, s3, s4, s5, s6]
    _, Lc = dice_coefficient(Gn[5], Sn[5], training_mask=training_mask)
    tf.summary.scalar("Lc_loss", Lc)

    one = tf.ones_like(Sn[5])
    zero = tf.zeros_like(Sn[5])
    W = tf.where(Sn[5] >= 0.5, x=one, y=zero)
    D = 0
    for i in range(5):
        di, _ = dice_coefficient(Gn[i] * W, Sn[i] * W, training_mask=training_mask)
        D += di
    Ls = 1 - D / 5.0
    tf.summary.scalar("Ls_loss", Ls)
    lambda_ = 0.7
    L = lambda_ * Lc + (1 - lambda_) * Ls
    return L

