import * as tf from '@tensorflow/tfjs';
export declare type MobileNetMultiplier = 0.50 | 0.75 | 1.0 | 1.01;
export declare type ConvolutionType = 'conv2d' | 'separableConv';
export declare type ConvolutionDefinition = [ConvolutionType, number];
export declare type OutputStride = 32 | 16 | 8;
export declare function assertValidOutputStride(outputStride: any): void;
export declare function assertValidResolution(resolution: any, outputStride: number): void;
export declare function assertValidScaleFactor(imageScaleFactor: any): void;
export declare const mobileNetArchitectures: {
    [name: string]: ConvolutionDefinition[];
};
export declare class MobileNet {
    private variables;
    private convolutionDefinitions;
    private PREPROCESS_DIVISOR;
    private ONE;
    constructor(variables: {
        [varName: string]: tf.Tensor;
    }, convolutionDefinitions: ConvolutionDefinition[]);
    predict(input: tf.Tensor3D, outputStride: OutputStride): tf.Tensor<tf.Rank.R3>;
    convToOutput(mobileNetOutput: tf.Tensor3D, outputLayerName: string): tf.Tensor3D;
    private conv(inputs, stride, blockId);
    private separableConv(inputs, stride, blockID, dilations?);
    private weights(layerName);
    private biases(layerName);
    private depthwiseWeights(layerName);
    dispose(): void;
}
