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

import * as poseDetection from './index';
import {KARMA_SERVER} from './shared/test_util';

const SIMULATED_INTERVAL = 33.333;  // in milliseconds

export async function loadVideo(
    videoPath: string, videoFPS: number,
    callback: (video: poseDetection.PoseDetectorInput, timestamp: number) =>
        Promise<poseDetection.Pose[]>,
    expected: number[][][],
    model: poseDetection.SupportedModels): Promise<HTMLVideoElement> {
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
      const poses = await callback(video, simulatedTimestamp);

      const expectedKeypoints = expected[idx].map(([x, y]) => {
        return {x, y};
      });

      ctx.drawImage(video, 0, 0);
      draw(expectedKeypoints, ctx, model, 'Green');

      if (poses.length > 0 && poses[0].keypoints != null) {
        draw(poses[0].keypoints, ctx, model, 'Red');
      }

      const nextTime = video.currentTime + actualInterval;
      if (nextTime < video.duration) {
        video.currentTime = nextTime;
        // We set the timestamp increment to 33333 microseconds to simulate
        // the 30 fps video input. We do this so that the filter uses the
        // same fps as the reference test.
        // https://github.com/google/mediapipe/blob/ecb5b5f44ab23ea620ef97a479407c699e424aa7/mediapipe/python/solution_base.py#L297
        simulatedTimestamp += SIMULATED_INTERVAL;
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

function drawKeypoint(
    keypoint: poseDetection.Keypoint, ctx: CanvasRenderingContext2D): void {
  const circle = new Path2D();
  circle.arc(
      keypoint.x, keypoint.y, 4 /* radius */, 0 /* startAngle */, 2 * Math.PI);
  ctx.fill(circle);
  ctx.stroke(circle);
}

function drawSkeleton(
    keypoints: poseDetection.Keypoint[], model: poseDetection.SupportedModels,
    ctx: CanvasRenderingContext2D, color: string) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;

  const pairs = poseDetection.util.getAdjacentPairs(model);
  for (const pair of pairs) {
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
    keypoints: poseDetection.Keypoint[], ctx: CanvasRenderingContext2D,
    model: poseDetection.SupportedModels, color: string) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;

  for (const keypoint of keypoints) {
    drawKeypoint(keypoint, ctx);
  }
  drawSkeleton(keypoints, model, ctx, color);
}
