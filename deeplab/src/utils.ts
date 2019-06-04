import * as tf from '@tensorflow/tfjs';
import {
    DeepLabInput,
    Label,
    SegmentationData,
    RawSegmentationMap,
    Color,
    Legend,
} from './types';
import config from './config';

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

export const createPascalColormap = (): Color[] => {
    const colormap = new Array(config['DATASET_MAX_ENTRIES']['PASCAL']);
    for (let idx = 0; idx < config['DATASET_MAX_ENTRIES']['PASCAL']; ++idx) {
        colormap[idx] = new Array(3);
    }
    for (let shift = 7; shift > 4; --shift) {
        const indexShift = 3 * (7 - shift);
        for (let channel = 0; channel < 3; ++channel) {
            for (
                let idx = 0;
                idx < config['DATASET_MAX_ENTRIES']['PASCAL'];
                ++idx
            ) {
                colormap[idx][channel] |=
                    ((idx >> (channel + indexShift)) & 1) << shift;
            }
        }
    }
    return colormap;
};

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

export async function processSegmentationMap(
    segmentationMapTensor: RawSegmentationMap
): Promise<SegmentationData> {
    const [height, width] = segmentationMapTensor.shape;
    const colormap = createPascalColormap();
    const channels = Array<tf.TensorBuffer<tf.Rank, 'int32'>>(3).fill(
        tf.buffer(segmentationMapTensor.shape, 'int32')
    );
    const segmentationMapArray = (await segmentationMapTensor.array()) as number[][];
    const labels = new Set<Label>();
    for (let columnIndex = 0; columnIndex < height; ++columnIndex) {
        for (let rowIndex = 0; rowIndex < width; ++rowIndex) {
            const label: Label = segmentationMapArray[columnIndex][rowIndex];
            labels.add(label);
            colormap[label].forEach((depth, channel) => {
                channels[channel].set(depth, columnIndex, rowIndex);
            });
        }
    }
    const translatedSegmentationMapTensor = tf.tidy(() => {
        const channelTensors = channels.map(buffer => buffer.toTensor());
        const translatedSegmentationMapTensor = tf.stack(
            channelTensors,
            2
        ) as tf.Tensor3D;

        return translatedSegmentationMapTensor;
    });
    const segmentationMap = await tf.browser.toPixels(
        translatedSegmentationMapTensor
    );
    tf.dispose(translatedSegmentationMapTensor);

    const labelNames = config['LABELS'];
    const legend: Legend = Array.from(labels).reduce(
        (accumulator, label) => ({
            ...accumulator,
            [labelNames[label]]: colormap[label],
        }),
        {}
    );

    return [legend, segmentationMap];
}
