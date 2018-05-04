import * as tf from '@tensorflow/tfjs-core';

import {CheckpointLoader} from './checkpoint_loader';

export type ConvType = 'conv2d'|'seperableConv';
export type ConvolutionDefinition = [ConvType, number];
export type OutputStride = 32|16|8;

// clang-format off
const mobileNet100Architecture: ConvolutionDefinition[] = [
  ['conv2d', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1]
];

const mobileNet75Architecture: ConvolutionDefinition[]  = [
  ['conv2d', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1]
];

const mobileNet50Architecture: ConvolutionDefinition[]  = [
  ['conv2d', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 2],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1],
  ['seperableConv', 1]
];
// clang-format on

export const mobileNetArchitectures:
    {[name: string]: ConvolutionDefinition[]} = {
      100: mobileNet100Architecture,
      75: mobileNet75Architecture,
      50: mobileNet50Architecture
    }

type Layer = {
  blockId: number,
  stride: number,
  outputStride: number,
  convType: ConvType,
  rate: number
};

function toOutputStridedLayers(
    convolutionDefinition: ConvolutionDefinition[],
    outputStride: OutputStride): Layer[] {
  // The currentStride variable keeps track of the output stride of
  // the activations, i.e., the running product of convolution
  // strides up to the current network layer. This allows us to
  // invoke atrous convolution whenever applying the next
  // convolution would result in the activations having output
  // stride larger than the target outputStride.
  let currentStride = 1;

  // The atrous convolution rate parameter.
  let rate = 1;

  return convolutionDefinition.map(([convType, stride], blockId): Layer => {
    let layerStride, layerRate;

    if (currentStride === outputStride) {
      // If we have reached the target outputStride, then we need to
      // employ atrous convolution with stride=1 and multiply the atrous
      // rate by the current unit's stride for use in subsequent layers.
      layerStride = 1;
      layerRate = rate;
      rate *= stride;
    } else {
      layerStride = stride;
      layerRate = 1;
      currentStride *= stride;
    }

    return {
      blockId, convType, stride: layerStride, rate: layerRate,
          outputStride: currentStride
    }
  });
}

export class MobileNet {
  private variables: {[varName: string]: tf.Tensor};
  private convolutionDefinitions: ConvolutionDefinition[];

  private PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
  private ONE = tf.scalar(1);

  async load(
      checkpointUrl: string,
      convolutionDefinitions: ConvolutionDefinition[]): Promise<void> {
    const checkpointLoader = new CheckpointLoader(checkpointUrl);

    // clean up old weights
    this.dispose();

    this.variables = await checkpointLoader.getAllVariables();
    this.convolutionDefinitions = convolutionDefinitions;
  }

  predict(input: tf.Tensor3D, outputStride: OutputStride) {
    // Normalize the pixels [0, 255] to be between [-1, 1].
    const preprocessedInput =
        tf.cast(input, 'float32').div(this.PREPROCESS_DIVISOR).sub(this.ONE) as
        tf.Tensor3D;

    const layers =
        toOutputStridedLayers(this.convolutionDefinitions, outputStride);

    return layers.reduce(
        (previousLayer: tf.Tensor3D,
         {blockId, stride, convType, rate}: Layer) => {
          if (convType === 'conv2d') {
            return this.conv(previousLayer, stride, blockId);
          } else if (convType === 'seperableConv') {
            return this.seperableConv(previousLayer, stride, blockId, rate);
          } else {
            throw Error('Unknown conv type of ' + convType);
          }
        },
        preprocessedInput);
  }

  public convToOutput(mobileNetOutput: tf.Tensor3D, outputLayerName: string):
      tf.Tensor3D {
    return mobileNetOutput.conv2d(this.weights(outputLayerName), 1, 'same')
               .add(this.biases(outputLayerName)) as tf.Tensor3D;
  }

  private conv(inputs: tf.Tensor3D, stride: number, blockId: number):
      tf.Tensor3D {
    return inputs
               .conv2d(
                   this.weights(`Conv2d_${String(blockId)}`), stride, 'same')
               .add(this.biases(`Conv2d_${String(blockId)}`))
               // relu6
               .clipByValue(0, 6) as tf.Tensor3D;
  }

  private seperableConv(
      inputs: tf.Tensor3D, stride: number, blockID: number,
      dilations = 1): tf.Tensor3D {
    const dwLayer = `Conv2d_${String(blockID)}_depthwise`;
    const pwLayer = `Conv2d_${String(blockID)}_pointwise`;

    const x1 = inputs
                   .depthwiseConv2D(
                       this.depthwiseWeights(dwLayer), stride, 'same', 'NHWC',
                       dilations)
                   .add(this.biases(dwLayer))
                   // relu6
                   .clipByValue(0, 6) as tf.Tensor3D;

    const x2 = x1.conv2d(this.weights(pwLayer), [1, 1], 'same')
                   .add(this.biases(pwLayer))
                   // relu6
                   .clipByValue(0, 6) as tf.Tensor3D;

    return x2;
  }

  private weights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/weights`] as tf.Tensor4D;
  }

  private biases(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/biases`] as tf.Tensor1D;
  }

  private depthwiseWeights(layerName: string) {
    return this.variables[`MobilenetV1/${layerName}/depthwise_weights`] as
        tf.Tensor4D;
  }

  dispose() {
    for (const varName in this.variables) {
      this.variables[varName].dispose();
    }
  }
}
