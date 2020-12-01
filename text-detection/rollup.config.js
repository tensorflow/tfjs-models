/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import json from 'rollup-plugin-json';
import node from 'rollup-plugin-node-resolve';
import {terser} from 'rollup-plugin-terser';
import typescript from 'rollup-plugin-typescript2';

const settings = {
  name: 'text-detection',
  preamble: `/**
 * @license
 * Copyright ${new Date().getFullYear()} Google LLC. All Rights Reserved.
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
 */`,
};

function minify() {
  return terser({
    output: {
      preamble: settings['preamble'],
    },
    ecma: 8,
    module: true,
    compress: true,
    mangle: {reserved: ['tf']},
  });
}

function config({plugins = [], output = {}}) {
  return {
    input: 'src/index.ts',
    plugins: [
      json({
        preferConst: true,
        indent: '  ',
        compact: true,
        namedExports: true,
      }),
      typescript({
        tsconfigOverride: {
          compilerOptions: {
            module: 'ES2015',
          },
        },
      }),
      node(),
      ...plugins,
    ],
    output: {
      banner: settings['preamble'],
      globals: {
        '@tensorflow/tfjs': 'tf',
      },
      ...output,
    },
    external: ['@tensorflow/tfjs'],
  };
}

export default [
  config({
    output: {
      format: 'umd',
      name: settings['name'],
      file: `dist/${settings['name']}.js`,
    },
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: settings['name'],
      file: `dist/${settings['name']}.min.js`,
    },
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'es',
      file: `dist/${settings['name']}.esm.js`,
    },
  }),
];
