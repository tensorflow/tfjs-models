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

export async function load(base: SemanticSegmentationBaseModel = 'pascal') {
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
    const semanticSegmentation = new SemanticSegmentation(base);
    await semanticSegmentation.load();
    return semanticSegmentation;
}

export class SemanticSegmentation {
    private modelPath: string;
    private model: tf.GraphModel;
    constructor(base: SemanticSegmentationBaseModel) {
        this.modelPath = `${config['BASE_PATH']}${base}/model.json`;
    }

    public async load() {
        this.model = await tf.loadGraphModel(this.modelPath);
    }

    public async predict(input: DeepLabInput): Promise<SegmentationMap> {
        const segmentationMapTensor = tf.tidy(() => {
            const data = toInputTensor(input);
            return tf.squeeze(this.model.execute(data) as tf.Tensor);
        }) as tf.Tensor2D;

        const [height, width] = segmentationMapTensor.shape;
        const segmentationMapData = toSegmentationMap(segmentationMapTensor);

        segmentationMapTensor.dispose();
        this.dispose();

        return [height, width, segmentationMapData];
    }

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
