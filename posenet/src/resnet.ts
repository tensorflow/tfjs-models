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
    // TODO: use output stride to convert the weights
    this.model = model;
    this.outputStride = outputStride;
    const inputShape = this.model.inputs[0].shape as [number, number, number, number];
    this.inputDimensions = [inputShape[1], inputShape[2]];
  }

  predict(input: tf.Tensor3D) {
    return tf.tidy(() => {
      const asFloat = toFloatIfInt(input);
      const asBatch = asFloat.expandDims(0);
      const [displacementFwd, displacementBwd, offsets, heatmaps] =
          this.model.predict(asBatch) as tf.Tensor<tf.Rank>[];

      // heatmaps
      const heatmaps_3d = heatmaps.squeeze() as tf.Tensor3D;
      // const resized_heatmaps =tf.image.resizeBilinear(
      //   heatmaps_3d, [513, 513], true);
      // const heatmapScores = (resized_heatmaps as tf.Tensor3D).sigmoid();
      const heatmapScores = (heatmaps_3d as tf.Tensor3D).sigmoid();

      // offsets
      const offsets_3d = offsets.squeeze() as tf.Tensor3D;
      const resized_offsets = offsets_3d;
      // const resized_offsets = tf.image.resizeBilinear(
      //   offsets_3d, [513, 513], true);

      // displacement forward
      const displacementFwd3d = displacementFwd.squeeze() as tf.Tensor3D;
      const resized_displacementFwd = displacementFwd3d;
      // const resized_displacementFwd = tf.image.resizeBilinear(
      //   displacementFwd3d, [513, 513], true);

      // displacement backward
      const displacementBwd3d = displacementBwd.squeeze() as tf.Tensor3D;
      const resized_displacementBwd = displacementBwd3d;
      // const resized_displacementBwd = tf.image.resizeBilinear(
      //   displacementBwd3d, [513, 513], true); 
 
      return {
        heatmapScores, offsets: resized_offsets as tf.Tensor3D,
            displacementFwd: resized_displacementFwd as tf.Tensor3D,
            displacementBwd: resized_displacementBwd as tf.Tensor3D
      }
      // const heatmapScores = (heatmaps as tf.Tensor3D).sigmoid();
      // return {
      //    heatmapScores, offsets: offsets as tf.Tensor3D,
      //        displacementFwd: displacementFwd as tf.Tensor3D,
      //        displacementBwd: displacementBwd as tf.Tensor3D
      // }
    });
  }

  dispose() {
    this.model.dispose();
  }
}
