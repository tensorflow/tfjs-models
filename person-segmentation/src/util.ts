import * as tf from '@tensorflow/tfjs';

import * as ops from './ops';
import {PersonSegmentationInput} from './types';

export function getInputTensorDimensions(input: PersonSegmentationInput):
    [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

export function toInputTensor(input: PersonSegmentationInput) {
  return input instanceof tf.Tensor ? input : tf.fromPixels(input);
}

export function cropAndResizeTo(
    input: PersonSegmentationInput, [targetHeight, targetWidth]: number[]) {
  const [height, width] = getInputTensorDimensions(input);
  const imageTensor = toInputTensor(input);

  const targetAspect = targetWidth / targetHeight;
  const aspect = width / height;

  let croppedW: number;
  let croppedH: number;

  if (aspect > targetAspect) {
    // crop width to get aspect
    croppedW = Math.round(height * targetAspect);
    croppedH = height;
  } else {
    croppedH = Math.round(width / targetAspect);
    croppedW = width;
  }

  const startCropTop = Math.floor((height - croppedH) / 2);
  const startCropLeft = Math.floor((width - croppedW) / 2);

  const resizedWidth = targetWidth;
  const resizedHeight = targetHeight;

  const croppedAndResized = tf.tidy(() => {
    const cropped = tf.slice3d(
        imageTensor, [startCropTop, startCropLeft, 0], [croppedH, croppedW, 3]);

    return cropped.resizeBilinear([resizedHeight, resizedWidth]);
  });

  return {
    croppedAndResized,
    resizedDimensions: [resizedHeight, resizedWidth],
    crop: [startCropTop, startCropLeft, croppedH, croppedW]
  };
}

export function resizeAndPadTo(
    input: PersonSegmentationInput, [targetH, targetW]: [number, number],
    flipHorizontal = false): {
  resizedAndPadded: tf.Tensor3D,
  paddedBy: [[number, number], [number, number]]
} {
  const [height, width] = getInputTensorDimensions(input);
  const imageTensor = toInputTensor(input);

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
    resizeW = Math.ceil(targetH / aspect);

    const padWidth = targetW - resizeW;
    padL = Math.floor(padWidth / 2);
    padR = targetW - (resizeW + padL);
    padT = 0;
    padB = 0;
  }

  const resizedAndPadded = tf.tidy(() => {
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
    canvas: HTMLCanvasElement, input: PersonSegmentationInput, mask: number[],
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
    canvas: HTMLCanvasElement, input: PersonSegmentationInput,
    partSegmentation: number[], partColors: Array<[number, number, number]>,
    partMapDarkening = 0.3, flipHorizontal = true) {
  const coloredPartImage = tf.tidy(() => {
    const image =
        flipHorizontal ? toInputTensor(input).reverse(1) : toInputTensor(input);

    return applyColorMap(image, partSegmentation, partColors, partMapDarkening);
  });
  await tf.toPixels(coloredPartImage, canvas);

  coloredPartImage.dispose();
}
