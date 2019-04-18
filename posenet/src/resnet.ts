import * as tf from '@tensorflow/tfjs';

function toFloatIfInt(input: tf.Tensor3D): tf.Tensor3D {
  return tf.tidy(() => {
    if (input.dtype === 'int32')
      return input.toFloat().add(tf.scalar(-122.0, 'float32'));
    return input;
  })
}

export default class ResNet {
  readonly model: tf.GraphModel
  readonly outputStride: number
  readonly inputDimensions: [number, number]

  constructor(model: tf.GraphModel, outputStride: number) {
    this.model = model;
    this.outputStride = outputStride;
    const inputShape = this.model.inputs[0].shape as [number, number, number, number];
    this.inputDimensions = [inputShape[1], inputShape[2]];
  }

  predict(input: tf.Tensor3D) {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [displacementFwd4d, displacementBwd4d, offsets4d, heatmaps4d] =
          this.model.predict(asBatch) as tf.Tensor<tf.Rank>[];

      const heatmaps = heatmaps4d.squeeze() as tf.Tensor3D;
      const heatmapScores = heatmaps.sigmoid();
      const offsets = offsets4d.squeeze() as tf.Tensor3D;
      const displacementFwd = displacementFwd4d.squeeze() as tf.Tensor3D;
      const displacementBwd = displacementBwd4d.squeeze() as tf.Tensor3D;
      
      return {
        heatmapScores, offsets: offsets as tf.Tensor3D,
            displacementFwd: displacementFwd as tf.Tensor3D,
            displacementBwd: displacementBwd as tf.Tensor3D
      }
    });
  }

  dispose() {
    this.model.dispose();
  }
}
