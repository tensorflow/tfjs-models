import sys
import argparse

import numpy as np

import tensorflow as tf


def group_weights(weights):
    """
    Group each layer weights together, initially all weights are dict of 'layer_name/layer_var': np.array

    Example:
        input:  {
                    ...: ...
                    'conv2d/kernel': <np.array>,
                    'conv2d/bias': <np.array>,
                    ...: ...
                }
        output: [..., [...], [<conv2d/kernel-weights>, <conv2d/bias-weights>], [...], ...]

    """

    out_weights = []

    previous_layer_name = ""
    group = []

    for k, v in weights.items():

        layer_name = "/".join(k.split("/")[:-1])

        if layer_name == previous_layer_name:
            group.append(v)
        else:
            if group:
                out_weights.append(group)

            group = [v]
            previous_layer_name = layer_name

    out_weights.append(group)
    return out_weights


def convert_tensorflow_model(
    model_name, model_ckpt, example_img="../demo/src/examples/pandah.jpg"
):
    """ Loads and saves a TensorFlow model. """
    image_files = [example_img]
    eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
    with tf.Graph().as_default(), tf.Session() as sess:
        images, _ = eval_ckpt_driver.build_dataset(
            image_files, [0] * len(image_files), False
        )
        eval_ckpt_driver.build_model(images, is_training=False)
        sess.run(tf.global_variables_initializer())
        eval_ckpt_driver.restore_model(sess, model_ckpt)
        global_variables = tf.global_variables()
        weights = dict()
        for variable in global_variables:
            try:
                weights[variable.name] = variable.eval()
            except:
                print(f"Skipping variable {variable.name}, an exception occurred")
        keys = list(weights.keys())
        print(keys[:10], keys[-10:])
        # print(group_weights(weights))
        # tf.train.Saver().save(sess, "tmp/model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TF model to Keras and save for easier future loading"
    )
    parser.add_argument(
        "--source", type=str, default="dist/src", help="source code path"
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
        default="pretrained_tensorflow/efficientnet-b0/",
        help="checkpoint file path",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="pretrained_pytorch/efficientnet-b0.pth",
        help="output PyTorch model file name",
    )
    args = parser.parse_args()

    sys.path.append(args.source)
    import eval_ckpt_main

    convert_tensorflow_model(args.model_name, args.tf_checkpoint)
