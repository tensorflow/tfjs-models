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

export class UnionFind {
  private _store: number[] = [];
  private nextLabel = 0;  // the next label in line

  public getComponent(node: number) {
    return this._store[node];
  }

  public makeLabel() {
    const currentLabel = this.nextLabel;
    this.nextLabel += 1;
    this._store.push(currentLabel);
    return currentLabel;
  }

  // Label all of the relatives of the current node, including itself, in sync
  // with the given root
  public setRoot(node: number, rootNode: number) {
    while (this._store[node] < node) {
      const nodeParent = this._store[node];
      this._store[node] = rootNode;
      node = nodeParent;
    }
    this._store[node] = rootNode;
  }

  // Find the root of the given node
  public findRoot(node: number) {
    while (this._store[node] < node) {
      node = this._store[node];
    }
    return node;
  }

  // Find the root of the node and compress the tree
  public find(node: number) {
    const root = this.findRoot(node);
    this.setRoot(node, root);
    return root;
  }

  // Merge the two trees containing nodes i and j and return the new root
  public union(firstNode: number, secondNode: number) {
    let root = this.findRoot(firstNode);
    if (firstNode !== secondNode) {
      const secondRoot = this.findRoot(secondNode);
      root = Math.min(root, secondRoot);
      this.setRoot(secondNode, root);
    }
    this.setRoot(firstNode, root);
    return root;
  }

  // Flatten the Union Find tree and relabel the components
  public flattenAndRelabel() {
    let currentLabel = 1;
    for (let idx = 1; idx < this._store.length; ++idx) {
      if (this._store[idx] < idx) {
        this._store[idx] = this._store[this._store[idx]];
      } else {
        this._store[idx] = currentLabel;
        currentLabel += 1;
      }
    }
    return currentLabel;
  }
}
