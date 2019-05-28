import * as tf from '@tensorflow/tfjs';
import { DeepLabInput } from './types';
import config from './settings';

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
    const image =
        input instanceof tf.Tensor ? input : tf.browser.fromPixels(input);
    const [height, width] = image.shape;
    const resizeRatio = config['CROP_SIZE'] / Math.max(width, height);
    const targetSize = [height, width].map(side =>
        Math.round(side * resizeRatio)
    );
    return image.resizeBilinear(targetSize as [number, number]);
}
