/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {Keypoint} from './calculators/interfaces/common_interfaces';

/** Karma server directory serving local files. */
export const KARMA_SERVER = './base/test_data';

export async function loadImage(
    imagePath: string, width: number,
    height: number): Promise<HTMLImageElement> {
  const img = new Image(width, height);
  const promise = new Promise<HTMLImageElement>((resolve, reject) => {
    img.crossOrigin = '';
    img.onload = () => {
      resolve(img);
    };
  });

  img.src = `${KARMA_SERVER}/${imagePath}`;

  return promise;
}

/**
 * Converts an RGBA image to a binary foreground mask based on an RGBA
 * threshold.
 *
 * @param image Input image to convert.
 *
 * @param r Minimum red value that denotes a foreground mask.
 * @param g Minimum green value that denotes a foreground mask.
 * @param b Minimum blue value that denotes a foreground mask.
 *
 * @return A boolean array of size number of pixels.
 */
export function imageToBooleanMask(
    rgbaData: Uint8ClampedArray, r: number, g: number, b: number) {
  const mask: boolean[] = [];
  for (let i = 0; i < rgbaData.length; i += 4) {
    mask.push(rgbaData[i] >= r && rgbaData[i + 1] >= g && rgbaData[i + 2] >= b);
  }
  return mask;
}

/**
 * Given two boolean masks, calculates the IOU percentage.
 *
 * @param image Input image to convert.
 *
 * @param expectedMask Expected mask values.
 * @param actualMask Actual mask values.
 *
 * @return A number denoting the IOU.
 */
export function segmentationIOU(
    expectedMask: boolean[], actualMask: boolean[]) {
  expect(expectedMask.length === actualMask.length);

  const sum = (mask: boolean[]) => mask.reduce((a, b) => a + +b, 0);

  const intersectionMask =
      expectedMask.map((value, index) => value && actualMask[index]);
  const iou = sum(intersectionMask) /
      (sum(expectedMask) + sum(actualMask) - sum(intersectionMask) +
       Number.EPSILON);

  return iou;
}

export async function loadVideo(
    videoPath: string, videoFPS: number,
    callback: (video: HTMLVideoElement, timestamp: number) =>
        Promise<Keypoint[]>,
    expected: number[][][], skeletonAdjacentPairs: number[][],
    simulatedInterval: number): Promise<HTMLVideoElement> {
  // We override video's timestamp with a fake timestamp.
  let simulatedTimestamp: number;
  // We keep a pointer for the expected array.
  let idx: number;

  const actualInterval = 1 / videoFPS;

  // Create a video element on the html page and serve the content through karma
  const video = document.createElement('video');
  // Hide video, and use canvas to render the video, so that we can also
  // overlay keypoints.
  video.style.visibility = 'hidden';
  const source = document.createElement('source');
  source.src = `${KARMA_SERVER}/${videoPath}`;
  source.type = 'video/mp4';
  video.appendChild(source);
  document.body.appendChild(video);
  const canvas = document.createElement('canvas');
  canvas.style.position = 'absolute';
  canvas.style.left = '0';
  canvas.style.top = '0';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const promise = new Promise<HTMLVideoElement>((resolve, reject) => {
    video.onseeked = async () => {
      const keypoints = await callback(video, simulatedTimestamp);

      const expectedKeypoints = expected[idx].map(([x, y]) => {
        return {x, y};
      });

      ctx.drawImage(video, 0, 0);
      draw(expectedKeypoints, ctx, skeletonAdjacentPairs, 'Green');

      draw(keypoints, ctx, skeletonAdjacentPairs, 'Red');

      const nextTime = video.currentTime + actualInterval;
      if (nextTime < video.duration) {
        video.currentTime = nextTime;
        simulatedTimestamp += simulatedInterval;
        idx++;
      } else {
        resolve(video);
      }
    };
  });

  video.onloadedmetadata = () => {
    video.currentTime = 0.001;
    simulatedTimestamp = 0;
    idx = 0;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    // Must set below two lines, otherwise video width and height are 0.
    video.width = videoWidth;
    video.height = videoHeight;
    // Must set below two lines, otherwise canvas has a different size.
    canvas.width = videoWidth;
    canvas.height = videoHeight;
  };

  return promise;
}

export function getXYPerFrame(result: number[][][]): number[][][] {
  return result.map(frameResult => {
    return frameResult.map(keypoint => [keypoint[0], keypoint[1]]);
  });
}

function drawKeypoint(keypoint: Keypoint, ctx: CanvasRenderingContext2D): void {
  const circle = new Path2D();
  circle.arc(
      keypoint.x, keypoint.y, 4 /* radius */, 0 /* startAngle */, 2 * Math.PI);
  ctx.fill(circle);
  ctx.stroke(circle);
}

function drawSkeleton(
    keypoints: Keypoint[], skeletonAdjacentPairs: number[][],
    ctx: CanvasRenderingContext2D, color: string) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  for (const pair of skeletonAdjacentPairs) {
    const [i, j] = pair;
    const kp1 = keypoints[i];
    const kp2 = keypoints[j];

    ctx.beginPath();
    ctx.moveTo(kp1.x, kp1.y);
    ctx.lineTo(kp2.x, kp2.y);
    ctx.stroke();
  }
}

function draw(
    keypoints: Keypoint[], ctx: CanvasRenderingContext2D,
    skeletonAdjacentPairs: number[][], color: string) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;

  for (const keypoint of keypoints) {
    drawKeypoint(keypoint, ctx);
  }
  drawSkeleton(keypoints, skeletonAdjacentPairs, ctx, color);
}
