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

class GetOffsetVectorsProgram implements tfwebgpu.WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  // A is heatmapScores, B is heatMapCoords.
  variableNames = ['A', 'B'];
  workgroupSize: [number, number, number];
  size = true;
  supportedLastDimension = 2;

  constructor(outputShape: number[]) {
    // Only 2d tensor whose last dimension is 2 is supported.
    if (outputShape.length !== 2 ||
        outputShape[1] !== this.supportedLastDimension) {
      throw new Error(`GetOffsetVectorsProgram only supports shape of [x, ${
          this.supportedLastDimension}], but current shape is ${outputShape}`);
    }
    const workgroupSizeX = 32;
    this.workgroupSize = [workgroupSizeX, 1, 1];
    this.outputShape = outputShape;
    const computeDispatchInfo = [outputShape[0], 1];
    this.dispatchLayout =
        tfwebgpu.webgpu_util.flatDispatchLayout(computeDispatchInfo);
    this.dispatch = tfwebgpu.webgpu_util.computeDispatch(
        this.dispatchLayout, computeDispatchInfo, this.workgroupSize);
    this.shaderKey = 'GetOffsetVectors';
  }

  getUserCode(): string {
    return `
    fn getOffsetPoint(y: i32, x: i32, index: i32) -> vec2<i32> {
      let outIndexY = y * uniforms.bShape.x * uniforms.bShape.y + x * uniforms.bShape.y + index;
      let outIndexX = outIndexY + uniforms.bShape.z;
      let outY = i32(B[outIndexY]);
      let outX = i32(B[outIndexX]);
      return vec2<i32>(outY, outX);
    }

    ${getMainHeaderString('index')} {
      if (index < uniforms.size) {
        let indexY = index * ${this.supportedLastDimension};
        let indexX = indexY + 1;
        let heatmapY = A[indexY];
        let heatmapX = A[indexX];
        let out = getOffsetPoint(i32(heatmapY), i32(heatmapX), index);
        result[indexY] = f32(out[0]);
        result[indexX] = f32(out[1]);
      }
    }
    `;
  }
}

export function getOffsetVectorsWebGPU<T extends tf.Tensor>(a: T, b: T): T {
  const webgpuBackend = tf.backend() as tfwebgpu.WebGPUBackend;
  const program = new GetOffsetVectorsProgram(a.shape);

  const outInfo: tf.TensorInfo =
      webgpuBackend.runWebGPUProgram(program, [a, b], 'float32');
  const value = tf.engine().makeTensorFromTensorInfo(outInfo) as T;

  return value;
}
