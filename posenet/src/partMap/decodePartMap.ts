/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {removePaddingAndResizeBack, resize2d} from '../util';

/**
 * Takes the sigmoid of the part heatmap output and generates a 2d one-hot
 * tensor with ones where the part's score has the maximum value.
 *
 * @param partHeatmapScores
 */
function toFlattenedOneHotPartMap(partHeatmapScores: tf.Tensor3D) {
  const [, , numParts] = partHeatmapScores.shape;
  const partMapLocations = partHeatmapScores.argMax(2);

  const partMapFlattened = partMapLocations.reshape([-1]) as tf.Tensor1D;

  return tf.oneHot(partMapFlattened, numParts);
}

/**
 * Takes the sigmoid of the part heatmap output and a list of rgb colors indexed
 * by part channel id, and generates a 3d tensor of an image with the
 * corresponding color at each pixel for the part with the highest value.
 * @param partHeatmapScores A 3d-tensor of the part heatmap output. The third
 * dimension corresponds to the part.
 *
 * @param partColors An array of rgb color values indexed by part channel id
 *
 * @returns A 3d tensor of an image with the corresponding color at each pixel
 * for the part with the highest value. Its height and width are the
 * output-strided size of the outputs.
 */
function decodeColoredPartMap(
    partHeatmapScores: tf.Tensor3D,
    partColors: Array<[number, number, number]>): tf.Tensor3D {
  const [partMapHeight, partMapWidth, numParts] = partHeatmapScores.shape;
  const flattenedOneHotPartMap = toFlattenedOneHotPartMap(partHeatmapScores);

  const colors = tf.tensor2d(partColors, [numParts, 3], 'int32');

  const coloredPartMapFlattened =
      flattenedOneHotPartMap.matMul(colors).cast('int32');

  return coloredPartMapFlattened.reshape([partMapHeight, partMapWidth, 3]);
}

function clipByMask(image: tf.Tensor3D, mask: tf.Tensor2D): tf.Tensor3D {
  return image.mul(mask.expandDims(2)) as tf.Tensor3D;
}

function toMask(segmentScores: tf.Tensor2D, threshold: number): tf.Tensor2D {
  return segmentScores.greater(tf.scalar(threshold)).cast('int32') as
      tf.Tensor2D;
}

/**
 * Takes the sigmoid of the segmentation mask and part heatmap output, and a
 * list of rgb colors indexed by part channel id, and generates a segmentation
 * mask and 3d tensor of an image with the corresponding color at each pixel for
 * the part with the highest value. The color values of the image are clipped by
 * the segmentation mask.
 * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
 * @param partHeatmapScores A 3d-tensor of the sigmoid of the part heatmap
 * output. The third dimension corresponds to the part.
 * @param segmentationThreshold The minimum that segmentation values must have
 * to be considered part of the person.  Affects the generation of the
 * segmentation mask and the clipping of the colored part image.
 *
 * @param partColors An array of rgb color values indexed by part channel id
 *
 * @returns A segmentatino mask, and a 3d tensor of an image with the
 * corresponding color at each pixel for the part with the highest value. The
 * color values of the image are clipped by the segmentation mask.  Both tensors
 * returned are resized and cropped to the original image's width and height.
 */
export function decodeAndScaleSegmentationAndPartMap(
    segmentScores: tf.Tensor3D, partHeatmapScores: tf.Tensor3D,
    [resizeH, resizeW]: [number, number],
    [[padT, padB], [padL, padR]]: number[][],
    [imageH, imageW]: [number, number], segmentationThreshold = 0.5,
    partColors: Array<[number, number, number]>) {
  return tf.tidy(() => {
    const scaledSegmentationScores =
        resize2d(segmentScores.squeeze(), [resizeH, resizeW], true);

    const scaledUpSegmentationScores = removePaddingAndResizeBack(
        scaledSegmentationScores, [imageH, imageW],
        [[padT, padB], [padL, padR]]);

    const scaledSegmentationMask =
        toMask(scaledUpSegmentationScores, segmentationThreshold);

    const scaledPartHeatmapScore =
        partHeatmapScores.resizeBilinear([imageH, imageW], true);

    const scaledUpPartHeatmapScore = removePaddingAndResizeBack(
        scaledPartHeatmapScore, [imageH, imageW], [[padT, padB], [padL, padR]]);

    const coloredPartMap =
        decodeColoredPartMap(scaledUpPartHeatmapScore, partColors);

    const clippedPartMap = clipByMask(coloredPartMap, scaledSegmentationMask);

    return {
      segmentationMask: scaledSegmentationMask,
      coloredPartImage: clippedPartMap
    };
  });
}
