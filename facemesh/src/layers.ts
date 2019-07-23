import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

function getExactlyOneTensor(xs: tf.Tensor[]) {
  let x;
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new Error(`Expected Tensor length to be 1; got ${xs.length}.`);
    }
    x = xs[0];
  } else {
    x = xs;
  }
  return x;
}

class ChannelPadding extends tfl.layers.Layer {
  private padding: number;
  // private mode: string;

  constructor(config: any) {
    super({});

    this.padding = config.padding;
    // this.mode = config.mode;
  }

  computeOutputShape(inputShape: number[]) {
    let batch = inputShape[0], dim1 = inputShape[1], dim2 = inputShape[2];
    let values = inputShape[3];
    let new_shape = [batch, dim1, dim2, values + this.padding];
    return new_shape;
  }

  call(inputs: tf.Tensor[]) {
    let input = getExactlyOneTensor(inputs);
    return tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, this.padding]], 0.0);
  }

  static get className() {
    return 'ChannelPadding';
  }
}

tf.serialization.registerClass(ChannelPadding);