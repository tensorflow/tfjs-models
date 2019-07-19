
/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {checkpoints} from './checkpoints';
import {decodePartSegmentation, toMask} from './decode_part_map';
import {assertValidOutputStride, MobileNet, MobileNetMultiplier, OutputStride} from './mobilenet';
import {ModelWeights} from './model_weights';
import {BodyPixInput, PartSegmentation, PersonSegmentation} from './types';
import {resizeAndPadTo, scaleAndCropToInputTensorShape, toInputTensor} from './util';

const segmentationModelImageDimensions: [number, number] = [353, 257];

export class BodyPix {
  mobileNet: MobileNet;

  constructor(mobileNet: MobileNet) {
    this.mobileNet = mobileNet;
  }

  predictForSegmentation(input: tf.Tensor3D, outputStride: OutputStride = 16):
      tf.Tensor3D {
    assertValidOutputStride(outputStride);

    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      return this.mobileNet.convToOutput(mobileNetOutput, 'segment_2')
          .sigmoid();
    });
  }

  predictForPartMap(input: tf.Tensor3D, outputStride: OutputStride = 16):
      {segmentScores: tf.Tensor3D, partHeatmapScores: tf.Tensor3D} {
    assertValidOutputStride(outputStride);
    return tf.tidy(() => {
      const mobileNetOutput = this.mobileNet.predict(input, outputStride);

      const segments =
          this.mobileNet.convToOutput(mobileNetOutput, 'segment_2');

      const partHeatmaps =
          this.mobileNet.convToOutput(mobileNetOutput, 'part_heatmap_2');

      return {
        segmentScores: segments.sigmoid(),
        partHeatmapScores: partHeatmaps.sigmoid()
      };
    });
  }

  /**
   * Given an image with a person, returns a binary array with 1 for the pixels
   * that are part of the person, and 0 otherwise. This does
   * standard ImageNet pre-processing before inferring through the model. Will
   * resize and crop the image to 353 x 257 while maintaining the original
   * aspect ratio before feeding through the network. The image pixels
   * should have values [0-255].
   *
   * @param input tf.Tensor3D
   * The input image to feed through the network.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the generation of the
   * segmentation mask.
   *
   * @return A 2d Tensor with 1 for the pixels that are part of the person,
   * and 0 otherwise. The width and height correspond to the same dimensions
   * of the input image.
   */
  estimatePersonSegmentationActivation(
      input: tf.Tensor3D, outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): tf.Tensor2D {
    assertValidOutputStride(outputStride);

    return tf.tidy(() => {
      const {
        resizedAndPadded,
        paddedBy,
      } = resizeAndPadTo(input, segmentationModelImageDimensions);

      const segmentScores =
          this.predictForSegmentation(resizedAndPadded, outputStride);

      const [resizedHeight, resizedWidth] = resizedAndPadded.shape;
      const [height, width] = input.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      return toMask(scaledSegmentScores.squeeze(), segmentationThreshold);
    });
  }

  /**
   * Given an image with a person, returns a binary array with 1 for the pixels
   * that are part of the person, and 0 otherwise. This does
   * standard ImageNet pre-processing before inferring through the model. Will
   * resize and crop the image to 353 x 257 while maintaining the original
   * aspect ratio before feeding through the network. The image pixels
   * should have values [0-255].
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the generation of the
   * segmentation mask.
   *
   * @return An object containing a width, height, and a binary array with 1 for
   * the pixels that are part of the person, and 0 otherwise. The array size
   * corresponds to the number of pixels in the image.  The width and height
   * correspond to the dimensions of the image the binary array is shaped to,
   * which are the same dimensions of the input image.
   */
  async estimatePersonSegmentation(
      input: BodyPixInput, outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): Promise<PersonSegmentation> {
    const segmentation = tf.tidy(() => {
      const imageTensor = toInputTensor(input);

      return this.estimatePersonSegmentationActivation(
          imageTensor, outputStride, segmentationThreshold);
    });

    const [height, width] = segmentation.shape;
    const result = await segmentation.data() as Uint8Array;

    segmentation.dispose();

    return {height, width, data: result};
  }

  /**
   * Given an image with a person, returns an array with a part id from 0-24 for
   * the pixels that are part of a corresponding body part, and -1 otherwise.
   * This does standard ImageNet pre-processing before inferring through the
   * model. Will resize and crop the image to 353 x 257 while maintaining the
   * original aspect ratio before feeding through the network. The image should
   * pixels should have values [0-255].
   *
   * @param input tf.Tensor3D
   * The input image to feed through the network.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the clipping of the colored
   * part image.
   *
   * @return A 2d Tensor with part ids from 0-24 for the pixels that are part of
   * a corresponding body part, and -1 otherwise. The width and height
   * correspond to the same dimensions of the input image.
   */
  estimatePartSegmentationActivation(
      input: tf.Tensor3D, outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): tf.Tensor2D {
    assertValidOutputStride(outputStride);

    return tf.tidy(() => {
      const {
        resizedAndPadded,
        paddedBy,
      } = resizeAndPadTo(input, segmentationModelImageDimensions);

      const {segmentScores, partHeatmapScores} =
          this.predictForPartMap(resizedAndPadded, outputStride);

      const [resizedHeight, resizedWidth] = resizedAndPadded.shape;

      const [height, width] = input.shape;
      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      const scaledPartHeatmapScore = scaleAndCropToInputTensorShape(
          partHeatmapScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      const segmentationMask =
          toMask(scaledSegmentScores.squeeze(), segmentationThreshold);

      return decodePartSegmentation(segmentationMask, scaledPartHeatmapScore);
    });
  }

  /**
   * Given an image with a person, returns an array with a part id from 0-24 for
   * the pixels that are part of a corresponding body part, and -1 otherwise.
   * This does standard ImageNet pre-processing before inferring through the
   * model. Will resize and crop the image to 353 x 257 while maintaining the
   * original aspect ratio before feeding through the network. The image should
   * pixels should have values [0-255].
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the clipping of the colored
   * part image.
   *
   * @return An object containing a width, height, and an array with a part id
   * from 0-24 for the pixels that are part of a corresponding body part, and -1
   * otherwise. The array size corresponds to the number of pixels in the image.
   * The width and height correspond to the dimensions of the image the array is
   * shaped to, which are the same dimensions of the input image.
   */
  async estimatePartSegmentation(
      input: BodyPixInput, outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): Promise<PartSegmentation> {
    const partSegmentation = tf.tidy(() => {
      const imageTensor = toInputTensor(input);

      return this.estimatePartSegmentationActivation(
          imageTensor, outputStride, segmentationThreshold);
    });

    const [height, width] = partSegmentation.shape;
    const data = await partSegmentation.data() as Int32Array;

    partSegmentation.dispose();

    return {height, width, data};
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

/**
 * Loads the BodyPix model instance from a checkpoint, with the
 * MobileNet architecture specified by the multiplier.
 *
 * @param multiplier An optional number with values: 1.01, 1.0, 0.75, or
 * 0.50. Defaults to 1.01. It is the float multiplier for the depth (number of
 * channels) for all convolution ops. The value corresponds to a MobileNet
 * architecture and checkpoint.  The larger the value, the larger the size of
 * the layers, and more accurate the model at the cost of speed.  Set this to
 * a smaller value to increase speed at the cost of accuracy.
 *
 */
export async function load(multiplier: MobileNetMultiplier = 0.75):
    Promise<BodyPix> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  // TODO: figure out better way to decide below.
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'number',
      () => `got multiplier type of ${typeof multiplier} when it should be a ` +
          `number.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier.toString()) >= 0,
      () => `invalid multiplier value of ${multiplier}` +
          `.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  const mobileNet = await mobilenetLoader.load(multiplier);

  return new BodyPix(mobileNet);
}

export const mobilenetLoader = {
  load: async(multiplier: MobileNetMultiplier): Promise<MobileNet> => {
    // TODO: move this into a config object, and use the multiplier to select it
    const checkpoint = checkpoints[multiplier];

    const baseUrl = checkpoint.url;

    const model = await tfconv.loadGraphModel(`${baseUrl}model.json`) as
        tfconv.GraphModel;

    const weights = new ModelWeights(model);

    return new MobileNet(weights, checkpoint.architecture);
  }
};
