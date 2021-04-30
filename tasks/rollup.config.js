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
import typescript from '@rollup/plugin-typescript';
import {uglify} from 'rollup-plugin-uglify';

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

function minify() {
  return uglify({output: {preamble: PREAMBLE}});
}

function config({plugins = [], output = {}, tsCompilerOptions = {}}) {
  const defaultTsOptions = {
    include: ['src/**/*.ts'],
    module: 'ES2015',
  };
  const tsoptions = Object.assign({}, defaultTsOptions, tsCompilerOptions);
  return {
    input: 'src/index.ts',
    plugins: [typescript(tsoptions), resolve(), commonjs(), ...plugins],
    output: {
      banner: PREAMBLE,
      sourcemap: true,
      ...output,
    },
  };
}

const packageName = 'tfTask';
export default [
  // node
  config({
    output: {format: 'cjs', name: packageName, file: 'dist/tfjs-tasks.node.js'},
    tsCompilerOptions: {target: 'es5'},
  }),
  // UMD ES5 unminified.
  config({
    output: {format: 'umd', name: packageName, file: 'dist/tfjs-tasks.js'},
    tsCompilerOptions: {target: 'es5'},
  }),
  // UMD ES5 minified.
  config({
    plugins: [minify()],
    output: {format: 'umd', name: packageName, file: 'dist/tfjs-tasks.min.js'},
    tsCompilerOptions: {target: 'es5'},
  }),
  // UMD ES2017 unminified.
  config({
    output:
        {format: 'umd', name: packageName, file: 'dist/tfjs-tasks.es2017.js'},
    tsCompilerOptions: {target: 'es2017'},
  }),
  // UMD ES2017 minified.
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: packageName,
      file: 'dist/tfjs-tasks.es2017.min.js'
    },
    tsCompilerOptions: {target: 'es2017'},
  }),
  // FESM ES2017 unminified.
  config({
    output: {format: 'es', name: packageName, file: 'dist/tfjs-tasks.fesm.js'},
    tsCompilerOptions: {target: 'es2017'},
  }),
  // FESM ES2017 minified.
  config({
    plugins: [minify()],
    output:
        {format: 'es', name: packageName, file: 'dist/tfjs-tasks.fesm.min.js'},
    tsCompilerOptions: {target: 'es2017'},
  }),
];
