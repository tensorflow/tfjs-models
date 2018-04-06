import * as tf from '@tensorflow/tfjs-core';

// does a floored division.  this is needed temporarily
// standard integer division can result in bugs
// https://github.com/PAIR-code/deeplearnjs/issues/847
function integerDiv(a: tf.Tensor1D, b: number) {
  const originalValues = a.buffer().values;
  const values = new Int32Array(a.shape[0]);

  for (let i = 0; i < a.shape[0]; i++) {
    values[i] = Math.floor(originalValues[i] / b);
  }

  return tf.tensor1d(values, 'int32');
}

function mod(a: tf.Tensor1D, b: number) {
  const floored = integerDiv(a, b);

  return a.sub(floored.mul(tf.scalar(b, 'int32')));
}

export function argmax2d(inputs: tf.Tensor3D): tf.Tensor2D {
  const [height, width, depth] = inputs.shape;

  const reshaped = inputs.reshape([height * width, depth]);
  const coords = reshaped.argMax(0) as tf.Tensor1D;

  const yCoords = integerDiv(coords, width).expandDims(1) as tf.Tensor2D;
  const xCoords = mod(coords, width).expandDims(1) as tf.Tensor2D;

  return tf.concat([yCoords, xCoords], 1);
}
