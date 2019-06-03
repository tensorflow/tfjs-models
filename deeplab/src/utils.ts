import * as tf from '@tensorflow/tfjs';
import config, { createPascalColormap } from './settings';
import { DeepLabInput } from './types';

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

export function toInputTensor(input: DeepLabInput) {
    return tf.tidy(() => {
        const image =
            input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
        const [height, width] = image.shape;
        const resizeRatio = config['CROP_SIZE'] / Math.max(width, height);
        const targetSize = [height, width].map(side =>
            Math.round(side * resizeRatio)
        );
        return tf.image
            .resizeBilinear(image, targetSize as [number, number])
            .expandDims(0);
    });
}

export async function toSegmentationMap(
    segmentationMapTensor: tf.Tensor2D
): Promise<Uint8ClampedArray> {
    const [height, width] = segmentationMapTensor.shape;
    const colormap = createPascalColormap();
    const channels = Array<tf.TensorBuffer<tf.Rank, 'int32'>>(3).fill(
        tf.buffer(segmentationMapTensor.shape)
    );
    const segmentationMap = (segmentationMapTensor.array() as Promise<
        number[][]
    >)
        .then(segmentationMapArray => {
            return tf.tidy(() => {
                for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
                    for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
                        colormap[
                            segmentationMapArray[columnIndex][rowIndex]
                        ].forEach((depth, channel) => {
                            channels[channel].set(depth, columnIndex, rowIndex);
                        });
                    }
                }

                const channelTensors = channels.map(buffer =>
                    buffer.toTensor()
                );
                const translatedSegmentationMapTensor = tf.concat(
                    channelTensors
                ) as tf.Tensor3D;

                return translatedSegmentationMapTensor;
            });
        })
        .then(async translatedSegmentationMapTensor => {
            const segmentationMap = await tf.browser.toPixels(
                translatedSegmentationMapTensor
            );
            translatedSegmentationMapTensor.dispose();
            return segmentationMap;
        });

    return segmentationMap;
}
