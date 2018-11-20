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

import * as tf from '@tensorflow/tfjs';

import {Example, SerializedDataset} from './types';

function getUID(): string {
  function s4() {
    return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
  }
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() +
      s4() + s4();
}

export class Dataset {
  private examples: {[id: string]: Example};

  constructor(artifacts?: SerializedDataset) {
    if (artifacts == null) {
      this.examples = {};
    } else {
      throw new Error('Not implemented yet');
    }
  }

  addExample(example: Example): string {
    tf.util.assert(example != null, 'Got null or undefined example');
    tf.util.assert(
        example.label != null && example.label.length > 0,
        `Expected label to be a non-empty string, ` +
            `but got ${JSON.stringify(example.label)}`);
    const uid = getUID();
    this.examples[uid] = example;
    return uid;
  }

  getExampleCounts(): {[label: string]: number} {
    const counts: {[label: string]: number} = {};
    for (const uid in this.examples) {
      const example = this.examples[uid];
      if (!(example.label in counts)) {
        counts[example.label] = 0;
      }
      counts[example.label]++;
    }
    return counts;
  }

  removeExample(uid: string): void {
    if (!(uid in this.examples)) {
      throw new Error(`Nonexisting example UID: ${uid}`);
    }
    delete this.examples[uid];
  }

  size(): number {
    return Object.keys(this.examples).length;
  }

  empty(): boolean {
    return this.size() === 0;
  }

  getVocabulary(): string[] {
    const vocab = new Set<string>();
    for (const uid in this.examples) {
      const example = this.examples[uid];
      vocab.add(example.label);
    }
    return [...vocab];
  }

  serialize(): SerializedDataset {
    return null;
  }
}
