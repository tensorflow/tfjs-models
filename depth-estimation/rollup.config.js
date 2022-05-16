/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import {terser} from 'rollup-plugin-terser';

const PREAMBLE = `/**
    * @license
    * Copyright ${(new Date).getFullYear()} Google LLC. All Rights Reserved.
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
    */`;

function config({plugins = [], output = {}, tsCompilerOptions = {}}) {
  const defaultTsOptions = {
    include: ['src/**/*.ts'],
    module: 'ES2015',
  };
  const tsoptions = Object.assign({}, defaultTsOptions, tsCompilerOptions);

  return {
    input: 'src/index.ts',
    plugins: [
      typescript(tsoptions), resolve(),
      commonjs({include: ['node_modules/@mediapipe/selfie_segmentation/**']}),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        '@tensorflow/tfjs-core': 'tf',
        '@tensorflow/tfjs-converter': 'tf',
        '@tensorflow-models/body-segmentation': 'bodySegmentation',
      },
      ...output,
    },
    external: [
      '@tensorflow/tfjs-core', '@tensorflow/tfjs-converter',
      '@tensorflow-models/body-segmentation'
    ]
  };
}

const packageName = 'depthEstimation';
export default [
  config({
    output: {format: 'umd', name: packageName, file: 'dist/depth-estimation.js'}
  }),
  config({
    plugins: [terser({output: {preamble: PREAMBLE, comments: false}})],
    output:
        {format: 'umd', name: packageName, file: 'dist/depth-estimation.min.js'}
  }),
  config({
    plugins: [terser({output: {preamble: PREAMBLE, comments: false}})],
    output: {format: 'es', file: 'dist/depth-estimation.esm.js'}
  })
];
