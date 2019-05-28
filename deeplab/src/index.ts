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
import { SemanticSegmentationBaseModel } from './types';
import config from './settings';

export default class SemanticSegmentation {
    private modelPath: string;
    private model: tf.GraphModel;
    constructor(base: SemanticSegmentationBaseModel) {
        if (['pascal', 'cityscapes', 'ade20k'].indexOf(base) === -1) {
            throw new Error(
                `SemanticSegmentation cannot be constructed ` +
                    `with an invalid base model ${base}. ` +
                    `Try one of 'pascal', 'cityscapes' and 'ade20k'.`
            );
        }
        this.modelPath = `${config['BASE_PATH']}${base}/model.json`;
    }
    public async load() {
        this.model = await tf.loadGraphModel(this.modelPath);
        return !!this.model;
    }
    public predict(X: any) {}

    /**
     * Dispose of the tensors allocated by the model.
     * You should call this when you are done with the model.
     */

    public dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
