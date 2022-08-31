/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

// import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';
// import * as tf from '@tensorflow/tfjs-core';

// export function rotate(
//     image: tf.Tensor4D, radians: number, fillValue: number[]|number,
//     center: [number, number]): tf.Tensor4D {
//   const imageShape = image.shape;
//   const imageHeight = imageShape[1];
//   const imageWidth = imageShape[2];
//   const sinFactor = Math.sin(radians);
//   const cosFactor = Math.cos(radians);

//   const centerX = Math.floor(
//       imageWidth * (typeof center === 'number' ? center : center[0]));
//   const centerY = Math.floor(
//       imageHeight * (typeof center === 'number' ? center : center[1]));

//   let fillSnippet = '';
//   if (typeof fillValue === 'number') {
//     fillSnippet = `float outputValue = ${fillValue.toFixed(2)};`;
//   } else {
//     fillSnippet = `
//       vec3 fill = vec3(${fillValue.join(',')});
//       float outputValue = fill[coords[3]];`;
//   }
//   const dispatchLayout =
//       tfwebgpu.webgpu.webgpu_util.flatDispatchLayout(imageShape);
//   const workGroupSize: [number, number, number] = [16, 1, 1];
//   const dispatch = tfwebgpu.webgpu.webgpu_util.computeDispatch(
//       dispatchLayout, imageShape, workGroupSize, [1, 1, 1]);

//   const program: tfwebgpu.webgpu.WebGPUProgram = {
//     variableNames: ['Image'],
//     outputShape: imageShape,
//     dispatchLayout,
//     dispatch,
//     workGroupSize,

//     userCode: `
//       void main() {
//         int index = int(gl_GlobalInvocationID.x);
//         ivec4 coords = getOutputCoords();
//         int x = coords[2];
//         int y = coords[1];
//         int coordX = int(float(x - ${centerX}) * ${cosFactor} -
//           float(y - ${centerY}) * ${sinFactor});
//         int coordY = int(float(x - ${centerX}) * ${sinFactor} +
//           float(y - ${centerY}) * ${cosFactor});
//         coordX = int(coordX + ${centerX});
//         coordY = int(coordY + ${centerY});

//         ${fillSnippet}

//         if(coordX > 0 && coordX < ${imageWidth} && coordY > 0 && coordY < ${
//         imageHeight}) {
//           outputValue = getImage(coords[0], coordY, coordX, coords[3]);
//         }

//       setOutput(index,outputValue);
//     }`
//   };

//   const webgpuBackend = tf.backend() as tfwebgpu.webgpu.WebGPUBackend;
//   return webgpuBackend.compileAndRun(program, [image]);
// }
