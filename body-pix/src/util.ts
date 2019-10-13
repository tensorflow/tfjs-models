import * as tf from '@tensorflow/tfjs-core';

import {BodyPixOutputStride} from './body_pix_model';
import {BodyPixInput, InputResolution, Padding} from './types';
import {Pose, TensorBuffer3D} from './types';

export function toValidInputResolution(
    inputResolution: number, outputStride: BodyPixOutputStride): number {
  if (isValidInputResolution(inputResolution, outputStride)) {
    return inputResolution;
  }

  return Math.floor(inputResolution / outputStride) * outputStride + 1;
}

export function validateInputResolution(inputResolution: InputResolution) {
  tf.util.assert(
      typeof inputResolution === 'number' ||
          typeof inputResolution === 'object',
      () => `Invalid inputResolution ${inputResolution}. ` +
          `Should be a number or an object with width and height`);

  if (typeof inputResolution === 'object') {
    tf.util.assert(
        typeof inputResolution.width === 'number',
        () => `inputResolution.width has a value of ${
            inputResolution.width} which is invalid; it must be a number`);
    tf.util.assert(
        typeof inputResolution.height === 'number',
        () => `inputResolution.height has a value of ${
            inputResolution.height} which is invalid; it must be a number`);
  }
}

export function getValidInputResolutionDimensions(
    inputResolution: InputResolution,
    outputStride: BodyPixOutputStride): [number, number] {
  validateInputResolution(inputResolution);
  if (typeof inputResolution === 'object') {
    return [
      toValidInputResolution(inputResolution.height, outputStride),
      toValidInputResolution(inputResolution.width, outputStride),
    ];
  } else {
    return [
      toValidInputResolution(inputResolution, outputStride),
      toValidInputResolution(inputResolution, outputStride),
    ];
  }
}

function isValidInputResolution(
    resolution: number, outputStride: number): boolean {
  return (resolution - 1) % outputStride === 0;
}

export function assertValidResolution(
    resolution: [number, number], outputStride: number) {
  tf.util.assert(
      typeof resolution[0] === 'number' && typeof resolution[1] === 'number',
      () => `both resolution values must be a number but had values ${
          resolution}`);

  tf.util.assert(
      isValidInputResolution(resolution[0], outputStride),
      () => `height of ${resolution[0]} is invalid for output stride ` +
          `${outputStride}.`);

  tf.util.assert(
      isValidInputResolution(resolution[1], outputStride),
      () => `width of ${resolution[1]} is invalid for output stride ` +
          `${outputStride}.`);
}

const VALID_OUTPUT_STRIDES = [8, 16, 32];
// tslint:disable-next-line:no-any
export function assertValidOutputStride(outputStride: any) {
  tf.util.assert(
      typeof outputStride === 'number', () => 'outputStride is not a number');
  tf.util.assert(
      VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0,
      () => `outputStride of ${outputStride} is invalid. ` +
          `It must be either 8, 16, or 32`);
}

export function getInputTensorDimensions(input: BodyPixInput):
    [number, number] {
  return input instanceof tf.Tensor ? [input.shape[0], input.shape[1]] :
                                      [input.height, input.width];
}

export function toInputTensor(input: BodyPixInput) {
  return input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
}

export function resizeAndPadTo(
    imageTensor: tf.Tensor3D, [targetH, targetW]: [number, number],
    flipHorizontal = false): {
  resizedAndPadded: tf.Tensor3D,
  paddedBy: [[number, number], [number, number]]
} {
  const [height, width] = imageTensor.shape;

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
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]],
    applySigmoidActivation = false): tf.Tensor3D {
  return tf.tidy(() => {
    let inResizedAndPadded = tensor.resizeBilinear(
        [resizedAndPaddedHeight, resizedAndPaddedWidth], true);

    if (applySigmoidActivation) {
      inResizedAndPadded = inResizedAndPadded.sigmoid();
    }

    return removePaddingAndResizeBack(
        inResizedAndPadded, [inputTensorHeight, inputTensorWidth],
        [[padT, padB], [padL, padR]]);
  });
}

export function removePaddingAndResizeBack(
    resizedAndPadded: tf.Tensor3D,
    [originalHeight, originalWidth]: [number, number],
    [[padT, padB], [padL, padR]]: [[number, number], [number, number]]):
    tf.Tensor3D {
  return tf.tidy(() => {
    return tf.image
        .cropAndResize(
            resizedAndPadded.expandDims(), [[
              padT / (originalHeight + padT + padB - 1.0),
              padL / (originalWidth + padL + padR - 1.0),
              (padT + originalHeight - 1.0) /
                  (originalHeight + padT + padB - 1.0),
              (padL + originalWidth - 1.0) / (originalWidth + padL + padR - 1.0)
            ]],
            [0], [originalHeight, originalWidth])
        .squeeze([0]);
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

export function padAndResizeTo(
    input: BodyPixInput, [targetH, targetW]: [number, number]):
    {resized: tf.Tensor3D, padding: Padding} {
  const [height, width] = getInputTensorDimensions(input);
  const targetAspect = targetW / targetH;
  const aspect = width / height;
  let [padT, padB, padL, padR] = [0, 0, 0, 0];
  if (aspect < targetAspect) {
    // pads the width
    padT = 0;
    padB = 0;
    padL = Math.round(0.5 * (targetAspect * height - width));
    padR = Math.round(0.5 * (targetAspect * height - width));
  } else {
    // pads the height
    padT = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padB = Math.round(0.5 * ((1.0 / targetAspect) * width - height));
    padL = 0;
    padR = 0;
  }

  const resized: tf.Tensor3D = tf.tidy(() => {
    let imageTensor = toInputTensor(input);
    imageTensor = tf.pad3d(imageTensor, [[padT, padB], [padL, padR], [0, 0]]);

    return imageTensor.resizeBilinear([targetH, targetW]);
  });

  return {resized, padding: {top: padT, left: padL, right: padR, bottom: padB}};
}

export async function toTensorBuffer<rank extends tf.Rank>(
    tensor: tf.Tensor<rank>,
    type: 'float32'|'int32' = 'float32'): Promise<tf.TensorBuffer<rank>> {
  const tensorData = await tensor.data();

  return tf.buffer(tensor.shape, type, tensorData as Float32Array) as
      tf.TensorBuffer<rank>;
}

export async function toTensorBuffers3D(tensors: tf.Tensor3D[]):
    Promise<TensorBuffer3D[]> {
  return Promise.all(tensors.map(tensor => toTensorBuffer(tensor, 'float32')));
}

export function scalePose(
    pose: Pose, scaleY: number, scaleX: number, offsetY = 0,
    offsetX = 0): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(({score, part, position}) => ({
                                    score,
                                    part,
                                    position: {
                                      x: position.x * scaleX + offsetX,
                                      y: position.y * scaleY + offsetY
                                    }
                                  }))
  };
}

export function scalePoses(
    poses: Pose[], scaleY: number, scaleX: number, offsetY = 0, offsetX = 0) {
  if (scaleX === 1 && scaleY === 1 && offsetY === 0 && offsetX === 0) {
    return poses;
  }
  return poses.map(pose => scalePose(pose, scaleY, scaleX, offsetY, offsetX));
}

export function flipPoseHorizontal(pose: Pose, imageWidth: number): Pose {
  return {
    score: pose.score,
    keypoints: pose.keypoints.map(
        ({score, part, position}) => ({
          score,
          part,
          position: {x: imageWidth - 1 - position.x, y: position.y}
        }))
  };
}

export function flipPosesHorizontal(poses: Pose[], imageWidth: number) {
  if (imageWidth <= 0) {
    return poses;
  }
  return poses.map(pose => flipPoseHorizontal(pose, imageWidth));
}

export function scaleAndFlipPoses(
    poses: Pose[], [height, width]: [number, number],
    [inputResolutionHeight, inputResolutionWidth]: [number, number],
    padding: Padding, flipHorizontal: boolean): Pose[] {
  const scaleY =
      (height + padding.top + padding.bottom) / (inputResolutionHeight);
  const scaleX =
      (width + padding.left + padding.right) / (inputResolutionWidth);

  const scaledPoses =
      scalePoses(poses, scaleY, scaleX, -padding.top, -padding.left);

  if (flipHorizontal) {
    return flipPosesHorizontal(scaledPoses, width);
  } else {
    return scaledPoses;
  }
}
