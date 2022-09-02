/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

/**
 * Execute all unit tests in the current directory. Takes a jasmine_util from
 * tfjs-core so that we use the tfjs-core module from the right test directory.
 */
// tslint:disable-next-line
export function runTests(jasmine_util: any): void {
  // tslint:disable-next-line:no-require-imports
  const jasmineCtor = require('jasmine');

  Error.stackTraceLimit = Infinity;

  process.on('unhandledRejection', e => {
    throw e;
  });

  jasmine_util.setTestEnvs([{name: 'test-cpu', backendName: 'cpu', flags: {}}]);

  const runner = new jasmineCtor();
  runner.loadConfig({spec_files: ['src/**/*_test.ts'], random: false});
  runner.execute();
}
