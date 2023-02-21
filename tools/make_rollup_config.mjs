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

import typescript from '@rollup/plugin-typescript';
import resolve from '@rollup/plugin-node-resolve';
import {terser} from 'rollup-plugin-terser';
import {visualizer} from 'rollup-plugin-visualizer';
import * as path from 'path';

const PREAMBLE = `/**
 * @license
 * Copyright ${(new Date).getFullYear()} Google LLC.
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
`;

const TS_OPTIONS = {
  include: ['src/**/*.ts'],
  module: 'ES2015',
};

export function makeRollupConfig({
  name,
  input = './src/index.ts',
  globals,
  outputDirectory = 'dist/',
}) {

  const allGlobals = {
    '@tensorflow/tfjs-core': 'tf',
    '@tensorflow/tfjs-converter': 'tf',
    ...globals
  };

  const configs = [];
  for (const format of ['umd', 'esm']) {
    for (const minify of [true, false]) {
      const dotMin = minify ? '.min' : '';
      const file = path.join(outputDirectory,
                             `${name}${dotMin}.${format}.js`);
      configs.push({
        input,
        plugins: [
          typescript(TS_OPTIONS),
          resolve(),
          ...(minify ? [terser()] : []),
          visualizer({
            sourcemap: true,
            filename: `${file}.stats.html`,
          }),
        ],
        output: {
          name, // For UMD
          file,
          format,
          banner: PREAMBLE,
          globals: allGlobals,
          sourcemap: true,
        },
        external: Object.keys(allGlobals),
      });
    }
  }
  return configs;
}
