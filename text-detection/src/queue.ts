/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export class Queue<T> {
  private _store: T[] = [];
  public push(val: T) {
    this._store.push(val);
  }
  public pop(): T|undefined {
    return this._store.shift();
  }
  public empty() {
    return this._store.length === 0;
  }
  public size() {
    return this._store.length;
  }
}
