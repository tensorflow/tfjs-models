import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

function getExactlyOneTensor(xs: tf.Tensor[]) {
  var x;
  if (Array.isArray(xs)) {
    if (xs.length !== 1) {
      throw new ValueError('Expected Tensor length to be 1; got ' + xs.length);
    }
    x = xs[0];
  } else {
    x = xs;
  }
  return x;
}

class ChannelPadding extends tfl.layers.Layer {
  static className = 'ChannelPadding';

  constructor(config) {
    super({});

    this.padding = config.padding;
    this.mode = config.mode;
  }

  computeOutputShape(inputShape) {
    let batch = inputShape[0], dim1 = inputShape[1], dim2 = inputShape[2];
    let values = inputShape[3];
    let new_shape = [batch, dim1, dim2, values + this.padding];
    return new_shape;
  }

  call(inputs, kwargs) {
    let input = getExactlyOneTensor(inputs);
    return tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, this.padding]], 0.0)
  }

  static get className() {
    return 'ChannelPadding';
  }
}

tf.serialization.registerClass(ChannelPadding);