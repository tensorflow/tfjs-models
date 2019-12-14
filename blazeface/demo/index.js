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

import * as blazeface from '@tensorflow-models/blazeface';

let model, ctx, videoWidth, videoHeight, video, canvas;

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': { facingMode: 'user' },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction =
  async () => {
    const prediction = await model.estimateFace(video);

    if (prediction) {
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
      for (let i = 0; i < prediction.length; i++) {
        const start = [prediction[i].left, prediction[i].top];
        const size = [prediction[i].width, prediction[i].height];
        ctx.fillStyle = "rgba(255, 0, 0, 0.5)";
        ctx.fillRect(start[0], start[1], size[0], size[1]);

        const landmarks = prediction[i].landmarks;

        ctx.fillStyle = "blue";
        for (let j = 0; j < landmarks.length; j++) {
          const x = landmarks[j][0];
          const y = landmarks[j][1];
          ctx.fillRect(x, y, 5, 5);
        }
      }
    }

    requestAnimationFrame(renderPrediction);
  }

const setupPage =
  async () => {
    await setupCamera();
    video.play();

    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;

    canvas = document.getElementById('output');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.fillStyle = "rgba(255, 0, 0, 0.5)";

    model = await blazeface.load();

    renderPrediction();
  }

setupPage();
