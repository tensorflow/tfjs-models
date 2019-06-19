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
import { config } from './config';
import {
  EfficientNetBaseModel,
  EfficientNetInput,
  // EfficientNetOutput,
} from './types';
import { toInputTensor } from './utils';

export class EfficientNet {
  private base: EfficientNetBaseModel;
  private model: Promise<tf.LayersModel>;
  private modelPath: string;
  constructor(base: EfficientNetBaseModel, isQuantized = true) {
    if (tf == null) {
      throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
          `also include @tensorflow/tfjs on the page before using this model.`
      );
    }
    if (['b0', 'b3', 'b5'].indexOf(base) === -1) {
      throw new Error(
        `EfficientNet cannot be constructed ` +
          `with an invalid base model ${base}. ` +
          `Try one of 'b0' or 'b3'.`
      );
    }
    this.base = base;

    this.modelPath = `${config['BASE_PATH']}/${
      isQuantized ? 'quantized/' : ''
    }${base}/model.json`;

    this.model = tf.loadLayersModel(this.modelPath);
  }

  public async predict(
    input: EfficientNetInput,
    canvas?: HTMLCanvasElement // : Promise<EfficientNetOutput>
  ) {
    const model = await this.model;

    return model.predict(toInputTensor(this.base, input));
  }

  /**
   * Dispose of the tensors allocated by the model.
   * You should call this when you are done with the model.
   */

  public async dispose() {
    (await this.model).dispose();
  }
}
