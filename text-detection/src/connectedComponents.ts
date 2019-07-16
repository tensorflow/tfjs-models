import {UnionFind} from './unionFind';

export function connectedComponents(graph: number[][]):
    {labelsCount: number, labels: number[][]} {
  const height = graph.length;
  const width = graph[0].length;
  const labels = Array.from(Array(height), () => new Array(width));
  const roster = new UnionFind();
  // make the label for the background
  roster.makeLabel();
  for (let row = 0; row < height; ++row) {
    for (let col = 0; col < width; ++col) {
      if (graph[row][col] > 0) {
        if (row > 0 && graph[row - 1][col] > 0) {
          if (col > 0 && graph[row][col - 1] > 0) {
            labels[row][col] =
                roster.union(labels[row][col - 1], labels[row - 1][col]);
          } else {
            labels[row][col] = labels[row - 1][col];
          }
        } else {
          if (col > 0 && graph[row][col - 1] > 0) {
            labels[row][col] = labels[row][col - 1];
          } else {
            labels[row][col] = roster.makeLabel();
          }
        }
      } else {
        labels[row][col] = 0;
      }
    }
  }
  const labelsCount = roster.flattenAndRelabel();
  for (let row = 0; row < height; ++row) {
    for (let col = 0; col < width; ++col) {
      const component = roster.find(labels[row][col]);
      labels[row][col] = component;
    }
  }
  return {labelsCount, labels};
}
