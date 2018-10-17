import * as tf from '@tensorflow/tfjs';

import {PosenetInput} from '../types';
import {toInputTensor} from '../util';

import * as ops from './ops';

export function applyDarkeningMask(
    image: tf.Tensor3D, mask: number[], darkLevel = 0.3): tf.Tensor3D {
  return tf.tidy(() => {
    const [height, width] = image.shape;

    const segmentationMask = tf.tensor2d(mask, [height, width], 'int32');

    const invertedMask = tf.scalar(1, 'int32').sub(segmentationMask);
    const darkeningMask = invertedMask.toFloat()
                              .mul(tf.scalar(darkLevel))
                              .add(segmentationMask.toFloat());

    return image.toFloat().mul(darkeningMask.expandDims(2)).toInt() as
        tf.Tensor3D;
  });
}


export async function maskAndDrawImageOnCanvas(
    canvas: HTMLCanvasElement, input: PosenetInput, mask: number[],
    darkLevel = 0.3, flipHorizontal = true) {
  const maskedImage = tf.tidy(() => {
    const image =
        flipHorizontal ? toInputTensor(input).reverse(1) : toInputTensor(input);

    return applyDarkeningMask(image, mask, darkLevel);
  });
  await tf.toPixels(maskedImage, canvas);

  maskedImage.dispose();
}

export function toColorMap(
    partMap: number[],
    partColors: Array<[number, number, number]>): tf.Tensor1D {
  const backend = tf.ENV.backend as tf.webgl.MathBackendWebGL;
  return tf.tidy(() => {
    const partMapTensor = tf.tensor1d(partMap);
    const partColorsTensor = tf.tensor2d(partColors, [partColors.length, 3]);

    const program = ops.toColorMap(partMapTensor.shape);

    return backend.compileAndRun(program, [partMapTensor, partColorsTensor]) as
        tf.Tensor1D;
  });
}

export function applyColorMap(
    image: tf.Tensor3D, partMap: number[],
    partColors: Array<[number, number, number]>,
    partMapDarkening = 0.3): tf.Tensor3D {
  return tf.tidy(() => {
    const darkenedImage = image.toFloat().mul(tf.scalar(partMapDarkening));

    const coloredPartMap = toColorMap(partMap, partColors);

    const partMapInImageShape =
        coloredPartMap.reshape([image.shape[0], image.shape[1], 3]);

    return darkenedImage
               .add(partMapInImageShape.toFloat().mul(
                   tf.scalar(1 - partMapDarkening)))
               .cast('int32') as tf.Tensor3D;
  });
}

export async function drawColoredPartImageOnCanvas(
    canvas: HTMLCanvasElement, input: PosenetInput, partSegmentation: number[],
    partColors: Array<[number, number, number]>, partMapDarkening = 0.3,
    flipHorizontal = true) {
  const coloredPartImage = tf.tidy(() => {
    const image =
        flipHorizontal ? toInputTensor(input).reverse(1) : toInputTensor(input);

    return applyColorMap(image, partSegmentation, partColors, partMapDarkening);
  });
  await tf.toPixels(coloredPartImage, canvas);

  coloredPartImage.dispose();
}
