export class UnionFind {
  private _store: number[] = [];
  private label = 0;  // the next label in line

  public makeLabel() {
    const nextLabel = this.label;
    this.label += 1;
    this._store.push(nextLabel);
    return nextLabel;
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
