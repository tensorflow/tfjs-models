/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
  return {
    input: 'index.js',
    plugins: [
      resolve(),
      // Polyfill require() from dependencies.
      commonjs({
        ignore: ['crypto', 'node-fetch', 'util'],
      }),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      globals: {
        'fs': 'fs',
        'path': 'path',
        'worker_threads': 'worker_threads',
        'perf_hooks': 'perf_hooks'
      },
      ...output,
    },
    external: [
      'crypto',
      'fs',
      'path',
      'worker_threads',
      'perf_hooks',
    ],
    onwarn: warning => {
      let {code} = warning;
      if (code === 'CIRCULAR_DEPENDENCY' || code === 'CIRCULAR' ||
          code === 'THIS_IS_UNDEFINED') {
        return;
      }
      console.warn('WARNING: ', warning.message);
    }
  };
}

const packageName = 'faceLandmarksDetectionDemo';
export default [config({
  plugins: [terser({
    output: {preamble: PREAMBLE, comments: false},
    compress: {typeofs: false},
  })],
  output: {format: 'umd', name: packageName, file: 'dist/index.js'}
})];
