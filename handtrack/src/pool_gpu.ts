/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

function getEffectiveFilterSize(filterSize: number, dilation: number) {
  if (dilation <= 1) {
    return filterSize;
  }

  return filterSize + (filterSize - 1) * (dilation - 1);
}

function parseTupleParam(param: number|number[]): [number, number, number] {
  if (typeof param === 'number') {
    return [param, param, param];
  }
  if (param.length === 2) {
    return [param[0], param[1], 1];
  }
  return param as [number, number, number];
}

function computePool2DInfo(
    inShape: [number, number, number, number],
    filterSize: [number, number]|number, strides: number|[number, number],
    dilations: number|[number, number], pad: 'same'|'valid'|number,
    roundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast') {
  const [filterHeight, filterWidth] = parseTupleParam(filterSize);

  let filterShape: [number, number, number, number];
  if (dataFormat === 'channelsLast') {
    filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
  } else if (dataFormat === 'channelsFirst') {
    filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  return computeConv2DInfo(
      inShape, filterShape, strides, dilations, pad, roundingMode, false,
      dataFormat);
}

function conditionalRound(
    value: number, roundingMode?: 'floor'|'round'|'ceil') {
  if (!roundingMode) {
    return value;
  }
  switch (roundingMode) {
    case 'round':
      // used for Caffe Conv
      return Math.round(value);
    case 'ceil':
      // used for Caffe Pool
      return Math.ceil(value);
    case 'floor':
      return Math.floor(value);
    default:
      throw new Error(`Unknown roundingMode ${roundingMode}`);
  }
}

function computeDefaultPad(
    inputShape: [number, number]|[number, number, number, number],
    fieldSize: number, stride: number, dilation = 1): number {
  const effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
  return Math.floor(
      (inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
}

function computeOutputShape2D(
    inShape: [number, number], fieldSize: number, stride: number,
    zeroPad?: number, roundingMode?: 'floor'|'round'|'ceil'): [number, number] {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  const inputRows = inShape[0];
  const inputCols = inShape[1];

  const outputRows = conditionalRound(
      (inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);

  const outputCols = conditionalRound(
      (inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);

  return [outputRows, outputCols];
}

function getPadAndOutInfo(
    pad: 'same'|'valid'|number, inHeight: number, inWidth: number,
    strideHeight: number, strideWidth: number, filterHeight: number,
    filterWidth: number, roundingMode?: 'floor'|'round'|'ceil') {
  let padInfo;
  let outHeight: number;
  let outWidth: number;

  if (typeof pad === 'number') {
    const padType = (pad === 0) ? 'VALID' : 'NUMBER';
    padInfo = {top: pad, bottom: pad, left: pad, right: pad, type: padType};
    const outShape = computeOutputShape2D(
        [inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
    outHeight = outShape[0];
    outWidth = outShape[1];
  } else if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    const padAlongHeight =
        Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
    const padAlongWidth =
        Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
    const top = Math.floor(padAlongHeight / 2);
    const bottom = padAlongHeight - top;
    const left = Math.floor(padAlongWidth / 2);
    const right = padAlongWidth - left;
    padInfo = {top, bottom, left, right, type: 'SAME'};
  } else if (pad === 'valid') {
    padInfo = {top: 0, bottom: 0, left: 0, right: 0, type: 'VALID'};
    outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
  } else {
    throw Error(`Unknown padding parameter: ${pad}`);
  }
  return {padInfo, outHeight, outWidth};
}

function computeConv2DInfo(
    inShape: [number, number, number, number],
    filterShape: [number, number, number, number],
    strides: number|[number, number], dilations: number|[number, number],
    pad: 'same'|'valid'|number, roundingMode?: 'floor'|'round'|'ceil',
    depthwise = false,
    dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast') {
  let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
  if (dataFormat === 'channelsLast') {
    [batchSize, inHeight, inWidth, inChannels] = inShape;
  } else if (dataFormat === 'channelsFirst') {
    [batchSize, inChannels, inHeight, inWidth] = inShape;
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  const [filterHeight, filterWidth, , filterChannels] = filterShape;
  const [strideHeight, strideWidth] = parseTupleParam(strides);
  const [dilationHeight, dilationWidth] = parseTupleParam(dilations);

  const effectiveFilterHeight =
      getEffectiveFilterSize(filterHeight, dilationHeight);
  const effectiveFilterWidth =
      getEffectiveFilterSize(filterWidth, dilationWidth);
  const {padInfo, outHeight, outWidth} = getPadAndOutInfo(
      pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight,
      effectiveFilterWidth, roundingMode);

  const outChannels = depthwise ? filterChannels * inChannels : filterChannels;

  let outShape: [number, number, number, number];
  if (dataFormat === 'channelsFirst') {
    outShape = [batchSize, outChannels, outHeight, outWidth];
  } else if (dataFormat === 'channelsLast') {
    outShape = [batchSize, outHeight, outWidth, outChannels];
  }

  return {
    batchSize,
    dataFormat,
    inHeight,
    inWidth,
    inChannels,
    outHeight,
    outWidth,
    outChannels,
    padInfo,
    strideHeight,
    strideWidth,
    filterHeight,
    filterWidth,
    effectiveFilterHeight,
    effectiveFilterWidth,
    dilationHeight,
    dilationWidth,
    inShape,
    outShape,
    filterShape
  };
}

export function maxPool(input: tf.Tensor4D, attrs: any) {
  const convInfo = computePool2DInfo(
      input.shape, attrs['ksize'], attrs['strides'], 1, attrs['padding']);

  const filterWidth = convInfo.filterWidth;
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;

  const ratio = attrs['strides'];
  const outputShape: any = [];
  input.shape.forEach((dim, idx) => {
    if (dim == null) {
      outputShape.push(null);
    } else {
      outputShape.push(Math.floor(dim / ratio[idx]));
    }
  });

  const padTop = convInfo.padInfo.top;
  const padLeft = convInfo.padInfo.left;
  // const outputShape = convInfo.outShape;

  const initializationValue = '-1.0 / 1e-20';

  const returnValue = `max(max(max(` +
      'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';

  const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
  const filterWidthVec4Remainder = filterWidth % 4;

  const updateSnippet = `
      minMaxValue = max(values, minMaxValue);
    `;

  const program: tf.webgl.GPGPUProgram = {
    variableNames: ['x'],
    outputShape,
    userCode: `
    const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
    const ivec2 pads = ivec2(${padTop}, ${padLeft});
    const float initializationValue = ${initializationValue};
    const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

    float count = 0.0;

    float getValue(int batch, int xR, int xC, int d) {
      if (xC < 0 || xC >= ${convInfo.inWidth}) {
        return initializationValue;
      }
      count += 1.0;
      return getX(batch, xR, xC, d);
    }

    void main() {
      ivec4 coords = getOutputCoords();
      int batch = coords[0];
      int d = coords[3];

      ivec2 xRCCorner = coords.yz * strides - pads;
      int xRCorner = xRCCorner.x;
      int xCCorner = xRCCorner.y;

      // max/min x(?, ?, d) to get y(yR, yC, d).
      // ? = to be determined
      vec4 minMaxValue = vec4(${initializationValue});
      float avgValue = 0.0;
      count = 0.0;

      for (int wR = 0; wR < ${effectiveFilterHeight};
          wR += ${dilationHeight}) {
        int xR = xRCorner + wR;

        if (xR < 0 || xR >= ${convInfo.inHeight}) {
          continue;
        }

        for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
          int xC = xCCorner + wC * ${dilationWidth};

          vec4 values = vec4(
            getValue(batch, xR, xC, d),
            getValue(batch, xR, xC + ${dilationWidth}, d),
            getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
            getValue(batch, xR, xC + 3 * ${dilationWidth}, d)
          );

          ${updateSnippet}
        }

        int xC = xCCorner + ${filterWidthNearestVec4};
        if (${filterWidthVec4Remainder === 1}) {
          vec4 values = vec4(
            getValue(batch, xR, xC, d),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${filterWidthVec4Remainder === 2}) {
          vec4 values = vec4(
            getValue(batch, xR, xC, d),
            getValue(batch, xR, xC + ${dilationWidth}, d),
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${filterWidthVec4Remainder === 3}) {
          vec4 values = vec4(
            getValue(batch, xR, xC, d),
            getValue(batch, xR, xC + ${dilationWidth}, d),
            getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
            initializationValue
          );

          ${updateSnippet}
        }
      }
      setOutput(${returnValue});
    }
  `
  };

  const webglBackend = tf.backend() as tf.webgl.MathBackendWebGL;
  return webglBackend.compileAndRun(program, [input]);
}

export function maxPoolWArgMax(input: tf.Tensor4D, attrs: any) {
  const convInfo = computePool2DInfo(
      input.shape, attrs['ksize'], attrs['strides'], 1, attrs['padding']);

  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const effectiveFilterHeight = convInfo.effectiveFilterHeight;
  const effectiveFilterWidth = convInfo.effectiveFilterWidth;

  const padTop = convInfo.padInfo.top;
  const padLeft = convInfo.padInfo.left;

  const ratio = attrs['strides'];
  let outputShape: any = [];
  input.shape.forEach((dim, idx) => {
    if (dim == null) {
      outputShape.push(null);
    } else {
      outputShape.push(Math.floor(dim / ratio[idx]));
    }
  });

  const program: tf.webgl.GPGPUProgram = {
    variableNames: ['x'],
    outputShape,
    userCode: `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        float minMaxValue = 0.0;
        float minMaxValueFound = 0.0;
        int minMaxPosition = 0;
        float avgValue = 0.0;

        for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${effectiveFilterWidth};
              wC += ${dilationWidth}) {
            int xC = xCCorner + wC;

            if (xC < 0 || xC >= ${convInfo.inWidth}) {
              continue;
            }

            float value = getX(batch, xR, xC, d);

            // If a min / max value has already been found, use it. If not,
            // use the current value.
            float currMinMaxValue = mix(
                value, minMaxValue, minMaxValueFound);
            if (value <= currMinMaxValue) {
              minMaxValue = value;
              minMaxValueFound = 1.0;
              minMaxPosition = wR * ${effectiveFilterWidth} + wC;
            }
          }
        }
        setOutput(float(minMaxPosition));
      }`
  };

  const webglBackend = tf.backend() as tf.webgl.MathBackendWebGL;
  return webglBackend.compileAndRun(program, [input]);
}
