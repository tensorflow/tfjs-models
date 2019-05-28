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

import * as tf from '@tensorflow/tfjs';
import { SemanticSegmentationBaseModel, DeepLabInput } from './types';
import config from './settings';
import { toInputTensor } from './utils';

export default class SemanticSegmentation {
    private modelPath: string;
    private model: Promise<tf.GraphModel>;
    constructor(base: SemanticSegmentationBaseModel) {
        if (['pascal', 'cityscapes', 'ade20k'].indexOf(base) === -1) {
            throw new Error(
                `SemanticSegmentation cannot be constructed ` +
                    `with an invalid base model ${base}. ` +
                    `Try one of 'pascal', 'cityscapes' and 'ade20k'.`
            );
        }
        this.modelPath = `${config['BASE_PATH']}${base}/model.json`;
        this.model = tf.loadGraphModel(this.modelPath);
    }

    public async predict(input: DeepLabInput) {
        const model = await this.model;
        const segmentationMap = tf.tidy(() => {
            const data = toInputTensor(input);
            const result = model.execute(data) as tf.Tensor;
            return result.dataSync() as Int32Array;
        });
        this.dispose();
        return segmentationMap;
    }

    /**
     * Dispose of the tensors allocated by the model.
     * You should call this when you are done with the model.
     */

    public dispose() {
        this.model.then(model => {
            model.dispose();
        });
    }
}
