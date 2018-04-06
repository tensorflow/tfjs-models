/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs-core';

import * as posenet from '../src';
import {Keypoint} from '../src';
import {OutputStride} from '../src/posenet';

function toTuple({y, x}: {y: number, x: number}) {
  return [y, x];
}

export function drawSegment(
    [ay, ax]: number[], [by, bx]: number[], color: string, scale: number,
    ctx: CanvasRenderingContext2D) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.strokeStyle = color;
  ctx.stroke();
}

export function drawSkeleton(
    keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D,
    scale = 1) {
  const adjacentKeyPoints =
      posenet.getAdjacentKeyPoints(keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints: Keypoint[]) => {
    drawSegment(
        toTuple(keypoints[0].point), toTuple(keypoints[1].point), '#0000ff',
        scale, ctx);
  });
}

export async function renderToCanvas(
    a: tf.Tensor3D, ctx: CanvasRenderingContext2D) {
  const [height, width, ] = a.shape;
  const imageData = new ImageData(width, height);
  const data = await a.data();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;
    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export function renderImageToCanvas(
    image: HTMLImageElement, size: [number, number],
    canvas: HTMLCanvasElement) {
  canvas.width = size[0];
  canvas.height = size[1];

  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}

export function drawHeatMapValues(
    heatMapValues: tf.Tensor2D, outputStride: 32|16|8,
    canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');

  const radius = 5;

  const scaledValues =
      heatMapValues.mul(tf.scalar(outputStride, 'int32')) as tf.Tensor2D;

  drawPoints(ctx, scaledValues, radius, 'blue');
}

function drawPoints(
    ctx: CanvasRenderingContext2D, points: tf.Tensor2D, radius: number,
    color: string) {
  const data = points.buffer().values;

  for (let i = 0; i < data.length; i += 2) {
    const pointY = data[i];
    const pointX = data[i + 1];

    if (pointX !== 0 && pointY !== 0) {
      ctx.beginPath();
      ctx.arc(pointX, pointY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = 'blue';
      ctx.fill();
    }
  }
}

export function drawOffsetVectors(
    heatMapValues: tf.Tensor2D, offsets: tf.Tensor3D,
    outputStride: OutputStride, scale = 1, ctx: CanvasRenderingContext2D) {
  const offsetPoints =
      posenet.singlePose.getOffsetPoints(heatMapValues, outputStride, offsets);
  const heatmapData = heatMapValues.buffer().values;
  const offsetPointsData = offsetPoints.buffer().values;

  for (let i = 0; i < heatmapData.length; i += 2) {
    const heatmapY = heatmapData[i] * outputStride;
    const heatmapX = heatmapData[i + 1] * outputStride;

    const offsetPointY = offsetPointsData[i];
    const offsetPointX = offsetPointsData[i + 1];

    drawSegment(
        [heatmapY, heatmapX], [offsetPointY, offsetPointX], '#ffff00', scale,
        ctx);
  }
}

export function drawKeypoints(
    keypoints: Keypoint[], minConfidence: number, ctx: CanvasRenderingContext2D,
    scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];
    if (keypoint.score < minConfidence) {
      continue;
    }

    const {y, x} = keypoint.point;

    ctx.beginPath();
    ctx.arc(x * scale, y * scale, 3, 0, 2 * Math.PI);
    ctx.fillStyle = 'blue';
    ctx.fill();
  }
}

export function drawBoundingBox(
    keypoints: Keypoint[], ctx: CanvasRenderingContext2D) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(
      boundingBox.minX, boundingBox.minY, boundingBox.maxX - boundingBox.minX,
      boundingBox.maxY - boundingBox.minY);
  ctx.stroke();
}
