/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as fs from 'fs';
import {join} from 'path';
import * as shell from 'shelljs';

// Exit if any commands error.
shell.set('-e');
process.on('unhandledRejection', e => {
  throw e;
});

const dir = '.';
const dirs = fs.readdirSync(dir)
                 .filter(f => fs.statSync(join(dir, f)).isDirectory())
                 .filter(f => !f.startsWith('.') && f !== 'node_modules');

dirs.forEach(dir => {
  console.log(`~~~~~~~~~~~~ Building ${dir} ~~~~~~~~~~~~`);

  shell.cd(dir);
  shell.exec('yarn');
  shell.exec('yarn build');

  const pkg = JSON.parse(fs.readFileSync('package.json').toString());
  if (pkg['scripts']['test'] != null) {
    console.log(`************ Testing ${dir} ************`);
    shell.exec('yarn test');
  } else {
    console.warn(
        `WARNING: ${dir} has no unit tests! ` +
        `Please consider adding unit tests to this model directory.`);
  }

  // Make sure peer dependencies and dev dependencies of tfjs match, and make
  // sure the version uses ^.
  const peerDeps = pkg.peerDependencies;
  const devDeps = pkg.devDependencies;
  if (peerDeps['@tensorflow/tfjs'] != null &&
      devDeps['@tensorflow/tfjs'] != null) {
    if (peerDeps['@tensorflow/tfjs'] != devDeps['@tensorflow/tfjs']) {
      throw new Error(
          `peerDependency version (${peerDeps['@tensorflow/tfjs']}) and ` +
          `devDependency version (${devDeps['@tensorflow/tfjs']}) of tfjs ` +
          `do not match for model ${dir}.`)
    }
  }
  if (peerDeps['@tensorflow/tfjs'] != null) {
    if (!peerDeps['@tensorflow/tfjs'].startsWith('^')) {
      throw new Error(
          `peerDependency version (${peerDeps['@tensorflow/tfjs']}) for ` +
          `${dir} must start with ^.`)
    }
  }
  if (devDeps['@tensorflow/tfjs'] != null) {
    if (!devDeps['@tensorflow/tfjs'].startsWith('^')) {
      throw new Error(
          `devDependency version (${peerDeps['@tensorflow/tfjs']}) for ${dir}` +
          `must start with ^.`)
    }
  }

  shell.cd('../');
  console.log();
  console.log();
});
