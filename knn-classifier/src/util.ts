import * as tf from '@tensorflow/tfjs';

export function concatWithNulls(
    ndarray1: tf.Tensor2D, ndarray2: tf.Tensor2D): tf.Tensor2D {
  if (ndarray1 == null && ndarray2 == null) {
    return null;
  }
  if (ndarray1 == null) {
    return ndarray2.clone();
  } else if (ndarray2 === null) {
    return ndarray1.clone();
  }
  return ndarray1.concat(ndarray2, 0);
}

export function topK(values: Float32Array, k: number):
    {values: Float32Array, indices: Int32Array} {
  const valuesAndIndices: Array<{value: number, index: number}> = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(k);
  const topkIndices = new Int32Array(k);
  for (let i = 0; i < k; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }
  return {values: topkValues, indices: topkIndices};
}
