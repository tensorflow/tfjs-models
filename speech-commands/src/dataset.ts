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

import {Example, SerializedExamples} from './types';

/**
 * Generate a pseudo-random UID.
 */
function getUID(): string {
  function s4() {
    return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
  }
  return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() +
      s4() + s4();
}

/**
 * A serializable, mutable set of speech/audio `Example`s;
 */
export class Dataset {
  private examples: {[id: string]: Example};
  private label2Ids: {[label: string]: string[]};

  /**
   * Constructor of `Dataset`.
   *
   * If called with no arguments (i.e., `artifacts` == null), an empty dataset
   * will be constructed.
   *
   * Else, the dataset will be deserialized from `artifacts`.
   *
   * @param artifacts Optional serialization artifacts to deserialize.
   */
  constructor(artifacts?: SerializedExamples) {
    if (artifacts == null) {
      this.examples = {};
      this.label2Ids = {};
    } else {
      // TODO(cais): Implement deserialization.
      throw new Error('Deserialization is not implemented yet');
    }
  }

  /**
   * Add an `Example` to the `Dataset`
   *
   * @param example A `Example`, with a label. The label must be a non-empty
   *   string.
   * @returns The UID for the added `Example`.
   */
  addExample(example: Example): string {
    tf.util.assert(example != null, 'Got null or undefined example');
    tf.util.assert(
        example.label != null && example.label.length > 0,
        `Expected label to be a non-empty string, ` +
            `but got ${JSON.stringify(example.label)}`);
    const uid = getUID();
    this.examples[uid] = example;
    if (!(example.label in this.label2Ids)) {
      this.label2Ids[example.label] = [];
    }
    this.label2Ids[example.label].push(uid);
    return uid;
  }

  /**
   * Get a map from `Example` label to number of `Example`s with the label.
   *
   * @returns A map from label to number of example counts under that label.
   */
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

  /**
   * Get all examples of a given label.
   *
   * @param label The requested label.
   * @return All examples of the given `label`.
   * @throws Error if label is `null` or `undefined`.
   */
  getExamples(label: string): Example[] {
    tf.util.assert(
       label != null,
       `Expected label to be a string, but got ${JSON.stringify(label)}`);
    if (!(label in this.label2Ids)) {
      throw new Error(`There are no examples of label "${label}"`);
    }
    return this.label2Ids[label].map(id => this.examples[id]);
  }

  /**
   * Remove an example from the `Dataset`.
   *
   * @param uid The UID of the example to remove.
   * @throws Error if the UID doesn't exist in the `Dataset`.
   */
  removeExample(uid: string): void {
    if (!(uid in this.examples)) {
      throw new Error(`Nonexisting example UID: ${uid}`);
    }
    delete this.examples[uid];
  }

  /**
   * Get the total number of `Example` currently held by the `Dataset`.
   *
   * @returns Total `Example` count.
   */
  size(): number {
    return Object.keys(this.examples).length;
  }

  /**
   * Query whether the `Dataset` is currently empty.
   *
   * I.e., holds zero examples.
   *
   * @returns Whether the `Dataset` is currently empty.
   */
  empty(): boolean {
    return this.size() === 0;
  }

  /**
   * Remove all `Example`s from the `Dataset`.
   */
  clear(): void {
    this.examples = {};
  }

  /**
   * Get the list of labels among all `Example`s the `Dataset` currently holds.
   *
   * @returns A sorted Array of labels, for the unique labels that belong to all
   *   `Example`s currently held by the `Dataset`.
   */
  getVocabulary(): string[] {
    const vocab = new Set<string>();
    for (const uid in this.examples) {
      const example = this.examples[uid];
      vocab.add(example.label);
    }
    const sortedVocab = [...vocab];
    sortedVocab.sort();
    return sortedVocab;
  }

  /**
   * Serialize the `Dataset`
   *
   * @returns A `SerializedDataset` object amenable to transmission and storage.
   */
  serialize(): SerializedExamples {
    return null;
  }
}
