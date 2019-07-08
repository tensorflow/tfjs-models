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
"""Model utilities."""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function


def build_learning_rate(
    initial_lr,
    global_step,
    steps_per_epoch=None,
    lr_decay_type="exponential",
    decay_factor=0.97,
    decay_epochs=2.4,
    total_steps=None,
    warmup_epochs=5,
):
    """Build learning rate."""
    if lr_decay_type == "exponential":
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True
        )
    elif lr_decay_type == "cosine":
        assert total_steps is not None
        lr = (
            0.5
            * initial_lr
            * (1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
        )
    elif lr_decay_type == "constant":
        lr = initial_lr
    else:
        assert False, "Unknown lr_decay_type : %s" % lr_decay_type

    if warmup_epochs:
        tf.logging.info("Learning rate warmup_epochs: %d" % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_lr = (
            initial_lr
            * tf.cast(global_step, tf.float32)
            / tf.cast(warmup_steps, tf.float32)
        )
        lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr


def build_optimizer(
    learning_rate, optimizer_name="rmsprop", decay=0.9, epsilon=0.001, momentum=0.9
):
    """Build optimizer."""
    if optimizer_name == "sgd":
        tf.logging.info("Using SGD optimizer")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optimizer_name == "momentum":
        tf.logging.info("Using Momentum optimizer")
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum
        )
    elif optimizer_name == "rmsprop":
        tf.logging.info("Using RMSProp optimizer")
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum, epsilon)
    else:
        tf.logging.fatal("Unknown optimizer:", optimizer_name)

    return optimizer


class TpuBatchNormalization(tf.compat.v1.layers.BatchNormalization):
    # class TpuBatchNormalization(tf.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError("TpuBatchNormalization does not support fused=True.")
        super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_shards_per_group):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        group_assignment = None
        if num_shards_per_group > 1:
            if num_shards % num_shards_per_group != 0:
                raise ValueError(
                    "num_shards: %d mod shards_per_group: %d, should be 0"
                    % (num_shards, num_shards_per_group)
                )
            num_groups = num_shards // num_shards_per_group
            group_assignment = [
                [x for x in range(num_shards) if x // num_shards_per_group == y]
                for y in range(num_groups)
            ]
        return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype
        )

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
            inputs, reduction_axes, keep_dims=keep_dims
        )

        num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
            num_shards_per_group = 1
        else:
            num_shards_per_group = max(8, num_shards // 8)
        tf.logging.info(
            "TpuBatchNormalization with num_shards_per_group %s", num_shards_per_group
        )
        if num_shards_per_group > 1:
            # Compute variance using: Var[X]= E[X^2] - E[X]^2.
            shard_square_of_mean = tf.math.square(shard_mean)
            shard_mean_of_square = shard_variance + shard_square_of_mean
            group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
            group_mean_of_square = self._cross_replica_average(
                shard_mean_of_square, num_shards_per_group
            )
            group_variance = group_mean_of_square - tf.math.square(group_mean)
            return (group_mean, group_variance)
        else:
            return (shard_mean, shard_variance)


def drop_connect(inputs, is_training, drop_connect_rate):
    """Apply drop connect."""
    if not is_training:
        return inputs

    # Compute keep_prob
    # TODO(tanmingxing): add support for training progress.
    keep_prob = 1.0 - drop_connect_rate

    # Compute drop_connect tensor
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.div(inputs, keep_prob) * binary_tensor
    return output


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
    """Archive a checkpoint if the metric is better."""
    ckpt_dir, ckpt_name = os.path.split(ckpt_path)

    saved_objective_path = os.path.join(ckpt_dir, "best_objective.txt")
    saved_objective = float("-inf")
    if tf.gfile.Exists(saved_objective_path):
        with tf.gfile.GFile(saved_objective_path, "r") as f:
            saved_objective = float(f.read())
    if saved_objective > ckpt_objective:
        tf.logging.info("Ckpt %s is worse than %s", ckpt_objective, saved_objective)
        return False

    filenames = tf.gfile.Glob(ckpt_path + ".*")
    if filenames is None:
        tf.logging.info("No files to copy for checkpoint %s", ckpt_path)
        return False

    # Clear the old folder.
    dst_dir = os.path.join(ckpt_dir, "archive")
    if tf.gfile.Exists(dst_dir):
        tf.gfile.DeleteRecursively(dst_dir)
    tf.gfile.MakeDirs(dst_dir)

    # Write checkpoints.
    for f in filenames:
        dest = os.path.join(dst_dir, os.path.basename(f))
        tf.gfile.Copy(f, dest, overwrite=True)
    ckpt_state = tf.train.generate_checkpoint_state_proto(
        dst_dir, model_checkpoint_path=ckpt_name, all_model_checkpoint_paths=[ckpt_name]
    )
    with tf.gfile.GFile(os.path.join(dst_dir, "checkpoint"), "w") as f:
        f.write(str(ckpt_state))
    with tf.gfile.GFile(os.path.join(dst_dir, "best_eval.txt"), "w") as f:
        f.write("%s" % ckpt_eval)

    # Update the best objective.
    with tf.gfile.GFile(saved_objective_path, "w") as f:
        f.write("%f" % ckpt_objective)

    tf.logging.info("Copying checkpoint %s to %s", ckpt_path, dst_dir)
    return True


# TODO(hongkuny): Consolidate this as a common library cross models.
class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.compat.v1.layers.Layer):
    """Wrap keras DepthwiseConv2D to tf.layers."""

    pass
