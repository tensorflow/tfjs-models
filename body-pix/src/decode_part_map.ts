/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs-core';

/**
 * Takes the sigmoid of the part heatmap output and generates a 2d one-hot
 * tensor with ones where the part's score has the maximum value.
 *
 * @param partHeatmapScores
 */
function toFlattenedOneHotPartMap(partHeatmapScores: tf.Tensor3D): tf.Tensor2D {
  const numParts = partHeatmapScores.shape[2];
  const partMapLocations = partHeatmapScores.argMax(2);

  const partMapFlattened = partMapLocations.reshape([-1]);

  return tf.oneHot(partMapFlattened, numParts) as tf.Tensor2D;
}

function clipByMask2d(image: tf.Tensor2D, mask: tf.Tensor2D): tf.Tensor2D {
  return image.mul(mask);
}

/**
 * Takes the sigmoid of the segmentation output, and generates a segmentation
 * mask with a 1 or 0 at each pixel where there is a person or not a person. The
 * segmentation threshold determines the threshold of a score for a pixel for it
 * to be considered part of a person.
 * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
 * @param segmentationThreshold The minimum that segmentation values must have
 * to be considered part of the person.  Affects the generation of the
 * segmentation mask and the clipping of the colored part image.
 *
 * @returns A segmentation mask with a 1 or 0 at each pixel where there is a
 * person or not a person.
 */
export function toMaskTensor(
    segmentScores: tf.Tensor2D, threshold: number): tf.Tensor2D {
  return tf.tidy(
      () =>
          (segmentScores.greater(tf.scalar(threshold)).toInt() as tf.Tensor2D));
}

/**
 * Takes the sigmoid of the person and part map output, and returns a 2d tensor
 * of an image with the corresponding value at each pixel corresponding to the
 * part with the highest value. These part ids are clipped by the segmentation
 * mask. Wherever the a pixel is clipped by the segmentation mask, its value
 * will set to -1, indicating that there is no part in that pixel.
 * @param segmentScores A 3d-tensor of the sigmoid of the segmentation output.
 * @param partHeatmapScores A 3d-tensor of the sigmoid of the part heatmap
 * output. The third dimension corresponds to the part.
 *
 * @returns A 2d tensor of an image with the corresponding value at each pixel
 * corresponding to the part with the highest value. These part ids are clipped
 * by the segmentation mask.  It will have values of -1 for pixels that are
 * outside of the body and do not have a corresponding part.
 */
export function decodePartSegmentation(
    segmentationMask: tf.Tensor2D,
    partHeatmapScores: tf.Tensor3D): tf.Tensor2D {
  const [partMapHeight, partMapWidth, numParts] = partHeatmapScores.shape;
  return tf.tidy(() => {
    const flattenedMap = toFlattenedOneHotPartMap(partHeatmapScores);
    const partNumbers = tf.range(0, numParts, 1, 'int32').expandDims(1);

    const partMapFlattened =
        flattenedMap.matMul(partNumbers as tf.Tensor2D).toInt();

    const partMap = partMapFlattened.reshape([partMapHeight, partMapWidth]);

    const partMapShiftedUpForClipping = partMap.add(tf.scalar(1, 'int32'));

    return clipByMask2d(
               partMapShiftedUpForClipping as tf.Tensor2D, segmentationMask)
        .sub(tf.scalar(1, 'int32'));
  });
}

export function decodeOnlyPartSegmentation(partHeatmapScores: tf.Tensor3D):
    tf.Tensor2D {
  const [partMapHeight, partMapWidth, numParts] = partHeatmapScores.shape;
  return tf.tidy(() => {
    const flattenedMap = toFlattenedOneHotPartMap(partHeatmapScores);
    const partNumbers = tf.range(0, numParts, 1, 'int32').expandDims(1);

    const partMapFlattened =
        flattenedMap.matMul(partNumbers as tf.Tensor2D).toInt();

    return partMapFlattened.reshape([partMapHeight, partMapWidth]);
  });
}
