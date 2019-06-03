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
import {
    SemanticSegmentationBaseModel,
    DeepLabInput,
    SegmentationMap,
} from './types';
import config from './settings';
import { toInputTensor, toSegmentationMap } from './utils';

export class SemanticSegmentation {
    private modelPath: string;
    private model: Promise<tf.GraphModel>;
    constructor(base: SemanticSegmentationBaseModel) {
        if (tf == null) {
            throw new Error(
                `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
                    `also include @tensorflow/tfjs on the page before using this model.`
            );
        }
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

    public async load() {}

    public async predict(input: DeepLabInput): Promise<SegmentationMap> {
        const model = await this.model;
        const segmentationMapTensor = tf.tidy(() => {
            const data = toInputTensor(input);
            return tf.squeeze(model.execute(data) as tf.Tensor);
        }) as tf.Tensor2D;

        const segmentationMapData = await toSegmentationMap(
            segmentationMapTensor
        );

        tf.dispose(segmentationMapTensor);
        this.dispose();

        return segmentationMapData;
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
