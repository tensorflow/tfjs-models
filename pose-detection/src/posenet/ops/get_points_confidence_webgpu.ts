/**
 * @license
 * Copyright 2023 Google LLC.
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

import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs-core';
import {getMainHeaderString} from './webgpu_util';

class GetpointsConfidenceProgram implements tfwebgpu.WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  // A is heatmapScores, B is heatmapValues.
  variableNames = ['A', 'B'];
  workgroupSize: [number, number, number];
  size = true;

  constructor(bShape: number[]) {
    const workgroupSizeX = 32;
    this.workgroupSize = [workgroupSizeX, 1, 1];
    this.outputShape = [bShape[0], 1];
    this.dispatchLayout =
        tfwebgpu.webgpu_util.flatDispatchLayout(this.outputShape);
    this.dispatch = tfwebgpu.webgpu_util.computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.shaderKey = 'getpointsConfidenceOp';
  }

  getUserCode(): string {
    return `
        ${getMainHeaderString('index')} {
          if (index < uniforms.size) {
            let y = B[index * 2];
            let x = B[index * 2 + 1];
            let outIndex = y * uniforms.aShape.x * uniforms.aShape.z + x * uniforms.aShape.z + index;
            result[index] = A[outIndex];
          }
        }
        `;
  }
}

export function getPointsConfidenceWebGPU<T extends tf.Tensor>(a: T, b: T): T {
  const webgpuBackend = tf.backend() as tfwebgpu.WebGPUBackend;
  const program = new GetpointsConfidenceProgram(b.shape);

  const outInfo: tf.TensorInfo =
      webgpuBackend.runWebGPUProgram(program, [a, b], 'float32');
  const value = tf.engine().makeTensorFromTensorInfo(outInfo) as T;

  return value;
}
