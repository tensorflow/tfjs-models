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

import config from './config.json';
import { Color } from './types';

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

export default config;
