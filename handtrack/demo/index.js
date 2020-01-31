/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import * as handtrack from '@tensorflow-models/handtrack';

const color = 'red';

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawKeypoints(ctx, keypoints) {
  let keypointsArray = keypoints.arraySync();

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(ctx, x, y, 3, color);
  }
}

function vectorDiff(v1, v2) {
  return [v1[0] - v2[0], v1[1] - v2[1]];
}

function vectorSum(v1, v2) {
  return [v1[0] + v2[0], v1[1] + v2[1]];
}

function rotatePoint(point, rad) {
  return [
    point[0] * Math.cos(rad) - point[1] * Math.sin(rad),
    point[0] * Math.sin(rad) + point[1] * Math.cos(rad)
  ];
}

function drawBox(ctx, box, angle, color) {
  const upperRight = [box[0], box[1]];
  const lowerLeft = [box[2], box[3]];
  const center = vectorSum(lowerLeft,
    vectorDiff(upperRight, lowerLeft).map(d => d / 2));

  let upperLeft = rotatePoint(vectorDiff(upperRight, center), Math.PI / 2);
  upperLeft = vectorSum(upperLeft, center);

  let lowerRight = rotatePoint(vectorDiff(lowerLeft, center), Math.PI / 2);
  lowerRight = vectorSum(lowerRight, center);

  const points = [upperRight, lowerRight, lowerLeft, upperLeft].map(point => {
    const rotated = rotatePoint(vectorDiff(point, center), angle);
    return vectorSum(rotated, center);
  });

  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  region.closePath();
  ctx.fillStyle = color;
  ctx.fill(region);
}

let model;

const statusElement = document.getElementById("status");
const status = msg => statusElement.innerText = msg;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

/**
 * Start the demo.
 */
const bindPage = async () => {
  model = await handtrack.load();
  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  landmarksRealTime(video);
}

const landmarksRealTime = async (video) => {
  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  const canvas = document.getElementById('output');

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  const ctx = canvas.getContext('2d');

  const cut = document.querySelector("#hand_cut");
  cut.width = 256;
  cut.height = 256;
  const cutCtx = cut.getContext('2d');

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  cutCtx.translate(256, 0);
  cutCtx.scale(-1, 1);

  async function frameLandmarks() {
    stats.begin();
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
    let result = await model.next_meshes(video);
    if (result) {
      drawKeypoints(ctx, result[0]);
      const angle = result[2];

      const box = result[3].startEndTensor.arraySync();
      drawBox(ctx, box[0], angle, `rgba(0, 0, 255, 0.2)`);

      const bbIncreased = result[4].startEndTensor.arraySync();
      drawBox(ctx, bbIncreased[0], angle, `rgba(255, 0, 0, 0.2)`);

      const bbSquared = result[5].startEndTensor.arraySync();
      drawBox(ctx, bbSquared[0], angle, `rgba(0, 255, 0, 0.2)`);

      const cutImage = result[1].arraySync();
      for(let r=0; r<256; r++) {
        for(let c=0; c<256; c++) {
          const point = cutImage[0][r][c];
          cutCtx.fillStyle = `rgb(${point.join(',')})`;
          cutCtx.fillRect(c, r, 1, 1);
        }
      }
    }
    stats.end();
    requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();
