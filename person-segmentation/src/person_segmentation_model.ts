
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

import * as tf from '@tensorflow/tfjs';

import {checkpoints} from './checkpoints';
import {decodePartSegmentation, toMask} from './decode_part_map';
import {assertValidOutputStride, MobileNet, MobileNetMultiplier, OutputStride} from './mobilenet';
import {ModelWeights} from './model_weights';
import {PersonSegmentationInput} from './types';
import {getInputTensorDimensions, resizeAndPadTo, scaleAndCropToInputTensorShape} from './util';

const segmentationModelImageDimensions: [number, number] = [353, 257];

export class PersonSegmentation {
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
   * Infer through PersonSegmentation, and segmentation for a single
   * person. This does standard ImageNet pre-processing before inferring through
   * the model. Will resize and crop the image to 353 x 257 while maintaining
   * the original aspect ratio before feeding through the network. The image
   * should pixels should have values [0-255]. This method returns an array with
   * values 0 or 1 corresponding to if a person was estimated to be in each
   * pixel. The array size corresponds to the number of pixels in the image. See
   * the readme for the body part ids.  If a value is 1, a person was estimated
   * to be in the corresponding pixel.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param flipHorizontal.  Defaults to false.  If the poses should be
   * flipped/mirrored  horizontally.  This should be set to true for videos
   * where the video is by default flipped horizontally (i.e. a webcam), and you
   * want the poses to be returned in the proper orientation.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the generation of the
   * segmentation mask and the clipping of the colored part image.
   *
   * @return An array with values 0 or 1 corresponding to if a person was
   * estimated to be in each pixel. The array size corresponds to the number of
   * pixels in the image. See the readme for the body part ids.  If a value is
   * 1, a person was estimated to be in the corresponding pixel.
   */
  async estimatePersonSegmentation(
      input: PersonSegmentationInput, flipHorizontal = false,
      outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): Promise<Uint8Array> {
    assertValidOutputStride(outputStride);

    const [height, width] = getInputTensorDimensions(input);

    const segmentation = tf.tidy(() => {
      const {
        resizedAndPadded,
        paddedBy,
      } =
          resizeAndPadTo(
              input, segmentationModelImageDimensions, flipHorizontal);

      const segmentScores =
          this.predictForSegmentation(resizedAndPadded, outputStride);

      const [resizedHeight, resizedWidth] = resizedAndPadded.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      const mask = toMask(scaledSegmentScores.squeeze(), segmentationThreshold);

      if (flipHorizontal) {
        return mask.reverse(1);
      } else {
        return mask;
      }
    });

    const result = await segmentation.data() as Uint8Array;

    return result;
  }

  /**
   * Infer through PersonSegmentation, and estimate body part segmentations for
   * a single person. This does standard ImageNet pre-processing before
   * inferring through the model. Will resize and crop the image to 353 x 257
   * while maintaining the original aspect ratio before feeding through the
   * network. The image should pixels should have values [0-255]. This method
   * returns an array with values 0-24 corresponding to the body part for each
   * pixel.  The array size corresponds to the number of pixels in the image.
   * See the readme for the body part ids.  If a value is -1, no body part was
   * estimated to be in that pixel.
   *
   * @param input ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
   * The input image to feed through the network.
   *
   * @param flipHorizontal.  Defaults to false.  If the poses should be
   * flipped/mirrored  horizontally.  This should be set to true for videos
   * where the video is by default flipped horizontally (i.e. a webcam), and you
   * want the poses to be returned in the proper orientation.
   *
   * @param outputStride the desired stride for the outputs.  Must be 32, 16,
   * or 8. Defaults to 16. The output width and height will be will be
   * (inputDimension - 1)/outputStride + 1
   *
   * @param segmentationThreshold The minimum that segmentation values must have
   * to be considered part of the person.  Affects the generation of the
   * segmentation mask and the clipping of the colored part image.
   *
   * @return An array with values 0-24 corresponding to the body part for each
   * pixel, indexed by pixel h, w.  The array size corresponds to the number of
   * pixels in the image. See the readme for the body part ids.  If a value is
   * -1, no body part was estimated to be in that pixel.
   */
  async estimatePartSegmentation(
      input: PersonSegmentationInput, flipHorizontal = false,
      outputStride: OutputStride = 16,
      segmentationThreshold = 0.5): Promise<Int32Array> {
    assertValidOutputStride(outputStride);

    const [height, width] = getInputTensorDimensions(input);

    const partSegmentation = tf.tidy(() => {
      const {
        resizedAndPadded,
        paddedBy,
      } =
          resizeAndPadTo(
              input, segmentationModelImageDimensions, flipHorizontal);

      const {segmentScores, partHeatmapScores} =
          this.predictForPartMap(resizedAndPadded, outputStride);

      const [resizedHeight, resizedWidth] = resizedAndPadded.shape;

      const scaledSegmentScores = scaleAndCropToInputTensorShape(
          segmentScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      const scaledPartHeatmapScore = scaleAndCropToInputTensorShape(
          partHeatmapScores, [height, width], [resizedHeight, resizedWidth],
          paddedBy);

      const segmentationMask =
          toMask(scaledSegmentScores.squeeze(), segmentationThreshold);

      const partSegmentation =
          decodePartSegmentation(segmentationMask, scaledPartHeatmapScore);

      if (flipHorizontal) {
        return partSegmentation.reverse(1);
      } else {
        return partSegmentation;
      }
    });

    const result = await partSegmentation.data() as Int32Array;

    partSegmentation.dispose();

    return result;
  }

  public dispose() {
    this.mobileNet.dispose();
  }
}

/**
 * Loads the PersonSegmentation model instance from a checkpoint, with the
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
    Promise<PersonSegmentation> {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  // TODO: figure out better way to decide below.
  const possibleMultipliers = Object.keys(checkpoints);
  tf.util.assert(
      typeof multiplier === 'number',
      `got multiplier type of ${typeof multiplier} when it should be a ` +
          `number.`);

  tf.util.assert(
      possibleMultipliers.indexOf(multiplier.toString()) >= 0,
      `invalid multiplier value of ${
          multiplier}.  No checkpoint exists for that ` +
          `multiplier. Must be one of ${possibleMultipliers.join(',')}.`);

  const mobileNet = await mobilenetLoader.load(multiplier);

  return new PersonSegmentation(mobileNet);
}

export const mobilenetLoader = {
  load: async(multiplier: MobileNetMultiplier): Promise<MobileNet> => {
    // TODO: move this into a config object, and use the multiplier to select it
    const checkpoint = checkpoints[multiplier];

    const baseUrl = checkpoint.url;

    const model = await tf.loadFrozenModel(
        `${baseUrl}tensorflowjs_model.pb`, `${baseUrl}weights_manifest.json`);

    const weights = new ModelWeights(model);

    return new MobileNet(weights, checkpoint.architecture);
  }
};
