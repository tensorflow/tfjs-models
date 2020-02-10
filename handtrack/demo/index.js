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

let videoWidth, videoHeight;
const color = 'red';

const MAX_KEYPOINTS = 30;
if(location.hash === '#debug') {
  for(let i=0; i<MAX_KEYPOINTS; i++) {
    const point = document.createElement("div");
    point.id = `point_${i}`;
    point.className = 'point';
    point.innerHTML = i;
    document.querySelector("#keypoints-wrapper").appendChild(point);
  }
}

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
    drawPoint(ctx, x - 2, y - 2, 3, color);

    if(location.hash === '#debug') {
      document.querySelector(`#point_${i}`).style.left = `${videoWidth - y}px`;
      document.querySelector(`#point_${i}`).style.top = `${x}px`;
    }
  }

  // Old model.
  const fingers = [
    [0, 1, 2, 3, 4], // thumb
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
  ];

  for(let i=0; i<fingers.length; i++) {
    const points = fingers[i].map(idx => keypointsArray[idx]);
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
      // Uncomment to test skeleton detection only
      // width: 256,
      // height: 256
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

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;

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
      document.querySelector("#keypoints-wrapper").className = 'show';

      let [keypoints, angle, cutImage, box, rotatedBox, shiftedBox, squaredBox, nextBox] = result;

      drawKeypoints(ctx, keypoints);

      if(box) {
        box = box.startEndTensor.arraySync();
        drawBox(ctx, box[0], angle, `rgba(0, 0, 255, 1)`); // blue
      }

      if(rotatedBox) {
        rotatedBox = rotatedBox.startEndTensor.arraySync();
        drawBox(ctx, rotatedBox[0], angle, `rgba(0, 0, 0, 1)`); // black
      }

      if(shiftedBox) {
        shiftedBox = shiftedBox.startEndTensor.arraySync();
        drawBox(ctx, shiftedBox[0], angle, `rgba(255, 0, 0, 1)`); // red
      }

      if(squaredBox) {
        squaredBox = squaredBox.startEndTensor.arraySync();
        drawBox(ctx, squaredBox[0], angle, `rgba(0, 255, 0, 1)`); // green
      }

      if(nextBox) {
        nextBox = nextBox.startEndTensor.arraySync();
        drawBox(ctx, nextBox[0], angle, `rgba(255, 0, 255, 1)`); // purple
      }

      if(cutImage) {
        const cutImageData = cutImage.arraySync();
        for(let r=0; r<256; r++) {
          for(let c=0; c<256; c++) {
            const point = cutImageData[0][r][c];
            cutCtx.fillStyle = `rgb(${point.join(',')})`;
            cutCtx.fillRect(c, r, 1, 1);
          }
        }
      }
    } else {
      document.querySelector("#keypoints-wrapper").className = 'hide';
    }
    stats.end();
    requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();
