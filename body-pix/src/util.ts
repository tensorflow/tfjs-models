import * as tf from '@tensorflow/tfjs';

import {BodyPixInput} from './types';

export function getInputTensorDimensions(input: BodyPixInput):
    [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

export function toInputTensor(input: BodyPixInput) {
  return input instanceof tf.Tensor ? input : tf.fromPixels(input);
}

export function resizeAndPadTo(
    input: BodyPixInput, [targetH, targetW]: [number, number],
    flipHorizontal = false): {
  resizedAndPadded: tf.Tensor3D,
  paddedBy: [[number, number], [number, number]]
} {
  const [height, width] = getInputTensorDimensions(input);

  const targetAspect = targetW / targetH;
  const aspect = width / height;

  let resizeW: number;
  let resizeH: number;
  let padL: number;
  let padR: number;
  let padT: number;
  let padB: number;

  if (aspect > targetAspect) {
    // resize to have the larger dimension match the shape.
    resizeW = targetW;
    resizeH = Math.ceil(resizeW / aspect);

    const padHeight = targetH - resizeH;
    padL = 0;
    padR = 0;
    padT = Math.floor(padHeight / 2);
    padB = targetH - (resizeH + padT);
  } else {
    resizeH = targetH;
    resizeW = Math.ceil(targetH * aspect);

    const padWidth = targetW - resizeW;
    padL = Math.floor(padWidth / 2);
    padR = targetW - (resizeW + padL);
    padT = 0;
    padB = 0;
  }

  const resizedAndPadded = tf.tidy(() => {
    const imageTensor = toInputTensor(input);
    // resize to have largest dimension match image
    let resized: tf.Tensor3D;
    if (flipHorizontal) {
      resized = imageTensor.reverse(1).resizeBilinear([resizeH, resizeW]);
    } else {
      resized = imageTensor.resizeBilinear([resizeH, resizeW]);
    }

    const padded = tf.pad3d(resized, [[padT, padB], [padL, padR], [0, 0]]);

    return padded;
  });

  return {resizedAndPadded, paddedBy: [[padT, padB], [padL, padR]]};
}

export function scaleAndCropToInputTensorShape(
    tensor: tf.Tensor3D,
    [inputTensorHeight, inputTensorWidth]: [number, number],
    [resizedAndPaddedHeight, resizedAndPaddedWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    tf.Tensor3D {
  return tf.tidy(() => {
    const inResizedAndPaddedSize = tensor.resizeBilinear(
        [resizedAndPaddedHeight, resizedAndPaddedWidth], true);

    return removePaddingAndResizeBack(
        inResizedAndPaddedSize, [inputTensorHeight, inputTensorWidth],
        [[padT, padB], [padL, padR]]);
  });
}

export function removePaddingAndResizeBack(
    resizedAndPadded: tf.Tensor3D,
    [originalHeight, originalWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    tf.Tensor3D {
  const [height, width] = resizedAndPadded.shape;
  // remove padding that was added
  const cropH = height - (padT + padB);
  const cropW = width - (padL + padR);

  return tf.tidy(() => {
    const withPaddingRemoved = tf.slice3d(
        resizedAndPadded as tf.Tensor3D, [padT, padL, 0],
        [cropH, cropW, resizedAndPadded.shape[2]]);

    const atOriginalSize = withPaddingRemoved.resizeBilinear(
        [originalHeight, originalWidth], true);

    return atOriginalSize;
  });
}

export function resize2d(
    tensor: tf.Tensor2D, resolution: [number, number],
    nearestNeighbor?: boolean): tf.Tensor2D {
  return tf.tidy(
      () => (tensor.expandDims(2) as tf.Tensor3D)
                .resizeBilinear(resolution, nearestNeighbor)
                .squeeze() as tf.Tensor2D);
}
