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
    describeWithFlags,
    NODE_ENVS,
} from '@tensorflow/tfjs-core/dist/jasmine_util';
import { SemanticSegmentation } from '.';

describeWithFlags('SemanticSegmentation', NODE_ENVS, () => {
    beforeAll(function() {
        jasmine.DEFAULT_TIMEOUT_INTERVAL = 20000;
    });
    it('SemanticSegmentation predict method should not leak', async () => {
        const model = new SemanticSegmentation('pascal');
        const x = tf.zeros([227, 500, 3]) as tf.Tensor3D;
        const numOfTensorsBefore = tf.memory().numTensors;

        await model.predict(x);
        await model.dispose();
        expect(tf.memory().numTensors).toEqual(numOfTensorsBefore);
    });
});
