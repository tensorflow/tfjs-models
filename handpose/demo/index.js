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

import * as handpose from '@tensorflow-models/handpose';

let videoWidth, videoHeight,
scatterGLHasInitialized = false, scatterGL;
const color = 'red';
const renderPointcloud = true;

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function drawKeypoints(ctx, keypoints) {
  const keypointsArray = keypoints;

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(ctx, x - 2, y - 2, 3, color);
  }

  const fingers = [
    [0, 1, 2, 3, 4], // thumb
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
  ];

  for(let i=0; i<fingers.length; i++) {
    const points = fingers[i].map(idx => keypoints[idx]);
    drawPath(ctx, points, 'red', false);
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

function drawPath(ctx, points, color, closePath) {
  ctx.strokeStyle = color;

  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if(closePath) {
    region.closePath();
  }
  ctx.stroke(region);
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

  drawPath(ctx, points, color, true);
}

let model;

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user'
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

const bindPage = async () => {
  model = await handpose.load();
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

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

  const canvas = document.getElementById('output');

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  const ctx = canvas.getContext('2d');

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  async function frameLandmarks() {
    stats.begin();
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
    const prediction = await model.estimateHand(video);
    if (prediction) {
      const result = prediction.landmarks;
      drawKeypoints(ctx, result);

      if (renderPointcloud === true && scatterGL != null) {
        // const pointsData = predictions.map(prediction => {
        //   let scaledMesh = prediction.scaledMesh;
        //   return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
        // });

        const pointsData = result.map(point => {
          return [-point[0], -point[1], -point[2]];
        });

        const dataset = new ScatterGL.Dataset(pointsData);

        if (!scatterGLHasInitialized) {
          scatterGL.render(dataset);
        } else {
          scatterGL.updateDataset(dataset);
        }
        scatterGLHasInitialized = true;
      }
    }
    stats.end();
    requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
      `width: ${500}px; height: ${500}px;`;

    scatterGL = new ScatterGL(
        document.querySelector('#scatter-gl-container'),
        {'rotateOnStart': false, 'selectEnabled': false});
  }
};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();
