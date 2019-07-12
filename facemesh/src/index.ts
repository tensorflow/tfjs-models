/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';
import {BlazePipeline} from './pipeline';

const BLAZEFACE_MODEL_URL =
    './blazeface_3x3_24ch_dd_bc5_rl6_256x256-web/tensorflowjs_model.pb';
const BLAZEFACE_WEIGHTS_URL =
    './blazeface_3x3_24ch_dd_bc5_rl6_256x256-web/weights_manifest.json';
const BLAZE_MESH_MODEL_PATH =
    './model_blaze__output_mesh_contours_faceflag-release-1.0.0-web-keras/model.json';

export class FaceMesh {
  private blazeFaceModel: tfconv.GraphModel;
  private blazeMeshModel: tfconv.GraphModel;
  private pipeline: any;

  async load() {
    const [blazeFaceModel, blazeMeshModel] =
        await Promise.all([this.loadFaceModel(), this.loadMeshModel()]);

    this.blazeFaceModel = blazeFaceModel;
    this.blazeMeshModel = blazeMeshModel;

    this.pipeline = new BlazePipeline(blazeFaceModel, blazeMeshModel);
  }

  loadFaceModel() {
    return tf.loadFrozenModel(BLAZEFACE_MODEL_URL, BLAZEFACE_WEIGHTS_URL);
  }

  loadMeshModel() {
    return tf.loadModel(BLAZE_MESH_MODEL_PATH);
  }

  getMesh(videoElement) {
    const image = tf.fromPixels(videoElement).toFloat();
    return pipeline.next_meshes(image);
  }
}