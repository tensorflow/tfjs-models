/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {EstimationConfig, InputResolution, ModelConfig, QuantBytes} from '../types';

export type PoseNetOutputStride = 32|16|8;
export type PoseNetArchitecture = 'ResNet50'|'MobileNetV1';
export type PoseNetDecodingMethod = 'single-person'|'multi-person';
export type MobileNetMultiplier = 0.50|0.75|1.0;

/**
 * Additional PoseNet model loading config.
 *
 * `architecture`: PoseNetArchitecture. It determines which PoseNet architecture
 * to load. The supported architectures are: MobileNetV1 and ResNet.
 *
 * `outputStride`: Specifies the output stride of the PoseNet model.
 * The smaller the value, the larger the output resolution, and more accurate
 * the model at the cost of speed.  Set this to a larger value to increase speed
 * at the cost of accuracy. Stride 32 is supported for ResNet and
 * stride 8,16,32 are supported for various MobileNetV1 models.
 *
 * * `inputResolution`: A number or an Object of type {width: number, height:
 * number}. Specifies the size the input image is scaled to before feeding it
 * through the PoseNet model.  The larger the value, more accurate the model at
 * the cost of speed. Set this to a smaller value to increase speed at the cost
 * of accuracy. If a number is provided, the input will be resized and padded to
 * be a square with the same width and height.  If width and height are
 * provided, the input will be resized and padded to the specified width and
 * height.
 *
 * `multiplier`: Optional. Options: 1.01, 1.0, 0.75, or 0.50.
 * The value is used only by MobileNet architecture. It is the float
 * multiplier for the depth (number of channels) for all convolution ops.
 * The larger the value, the larger the size of the layers, and more accurate
 * the model at the cost of speed. Set this to a smaller value to increase speed
 * at the cost of accuracy.
 *
 * `modelUrl`: Optional. An optional string that specifies custom url of the
 * model. This is useful for area/countries that don't have access to the model
 * hosted on GCP.
 *
 * `quantBytes`: Optional. Options: 1, 2, or 4.  This parameter affects weight
 * quantization in the models. The available options are
 * 1 byte, 2 bytes, and 4 bytes. The higher the value, the larger the model size
 * and thus the longer the loading time, the lower the value, the shorter the
 * loading time but lower the accuracy.
 */
export interface PosenetModelConfig extends ModelConfig {
  architecture: PoseNetArchitecture;
  outputStride: PoseNetOutputStride;
  inputResolution: InputResolution;
  multiplier?: MobileNetMultiplier;
  modelUrl?: string;
  quantBytes?: QuantBytes;
}

/**
 * Posenet Specific Inference Config
 *
 * `scoreThreshold`: For maxPoses > 1. Only return instance detections that have
 * root part score greater or equal to this value. Defaults to 0.5
 *
 * `nmsRadius`: For maxPoses > 1. Non-maximum suppression part distance in
 * pixels. It needs to be strictly positive. Two parts suppress each other if
 * they are less than `nmsRadius` pixels away. Defaults to 20.
 */
export interface PoseNetEstimationConfig extends EstimationConfig {
  scoreThreshold?: number;
  nmsRadius?: number;
}

export type Vector2D = {
  y: number,
  x: number
};

export declare type Part = {
  heatmapX: number,
  heatmapY: number,
  id: number
};

export declare type PartWithScore = {
  score: number,
  part: Part
};

export type Tuple<T> = [T, T];
export type StringTuple = Tuple<string>;
export type NumberTuple = Tuple<number>;
export interface NumberDict {
  [key: string]: number;
}
