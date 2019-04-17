/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

import {drawBoundingBox, drawKeypoints, drawSkeleton} from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
        'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
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

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
    showFeaturemaps: true,
  },
  net: null,
};

/**
 * Sets up dat.gui controller on the top-right of the window
 */
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({width: 300});

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  const algorithmController =
      gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  // network
  let input = gui.addFolder('Input');
  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  const architectureController = input.add(
      guiState.input, 'mobileNetArchitecture',
      ['1.01', '1.00', '0.75', '0.50']);
  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  input.add(guiState.input, 'outputStride', [8, 16, 32]);
  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
      .min(1)
      .max(20)
      .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);
  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.add(guiState.output, 'showFeaturemaps');
  output.open();


  architectureController.onChange(function(architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function(value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

/**
 * Sets up a frames per second panel on the top-left of the window
 */
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

/**
 * Feeds an image to posenet to estimate poses - this is where the magic
 * happens. This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  // since images are being fed from a webcam
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      guiState.net.dispose();

      // Load the ResNet50 PoseNet model
      guiState.net = await posenet.load(+guiState.changeToArchitecture);
      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;
    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
            video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);
        

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        // let heatmap_array = await guiState.net.estimateMultiplePoses(
        //     video, imageScaleFactor, flipHorizontal, outputStride,
        //     guiState.multiPoseDetection.maxPoseDetections,
        //     guiState.multiPoseDetection.minPartConfidence,
        //     guiState.multiPoseDetection.nmsRadius);

        // edges of parent kpt to child kpt.
        let edges = [[0, 1],
                     [1, 3],
                     [0, 2],
                     [2, 4],
                     [0, 5],
                     [5, 7],
                     [7, 9],
                     [5, 11],
                     [11, 13],
                     [13, 15],
                     [0, 6],
                     [6, 8],
                     [8, 10],
                     [6, 12],
                     [12, 14],
                     [14, 16]];

        let edges_bwd = [[1, 0],
                         [3, 1],
                         [2, 0],
                         [4, 2],
                         [5, 0],
                         [7, 5],
                         [9, 7],
                         [11, 5],
                         [13, 11],
                         [15, 13],
                         [6, 0],
                         [8, 6],
                         [10, 8],
                         [12, 6],
                         [14, 12],
                         [16, 14]];

        let outputs = await guiState.net.estimateMultiplePoses(
            video, imageScaleFactor, flipHorizontal, outputStride,
            guiState.multiPoseDetection.maxPoseDetections,
            guiState.multiPoseDetection.minPartConfidence,
            guiState.multiPoseDetection.nmsRadius);

        let heatmap_array = outputs[0];
        let offsets_array = outputs[1];
        let displacement_fwd_array = outputs[2];
        let displacement_bwd_array = outputs[3];
        let single_pose = outputs[4];
        let all_poses = outputs[5];

        // poses.push(single_pose);
        console.log('[visualize] num_poses_to_add', all_poses.length);
        poses = poses.concat(all_poses);
        console.log('[visualize] num_poses', poses.length);

        var canvas = document.getElementById('heatmap');
        var ctx2 = canvas.getContext('2d');
        var imageData = ctx2.createImageData(513, 513);
        for (let i = 0; i < imageData.height * imageData.width; i++) {
          imageData.data[4 * i + 0] = 0;
          imageData.data[4 * i + 1] = 0;
          imageData.data[4 * i + 2] = 0;
          imageData.data[4 * i + 3] = 255;
        }

        if (guiState.output.showFeaturemaps) {
          for (let i = 0; i < imageData.height * imageData.width; i++) {
            let max_p = 0;
            for (let k = 0; k < 17; k++) {
              let p = Math.round(255 * heatmap_array[17 * i + k]);
              if (p > max_p) {
                max_p = p;
              }
            }
            imageData.data[4 * i + 0] = max_p;
            imageData.data[4 * i + 1] = 0;
            imageData.data[4 * i + 2] = 0;
            imageData.data[4 * i + 3] = 255;
          }
        }

        ctx2.putImageData(imageData, 0, 0, 0, 0, 1000, 1000);

        if (guiState.output.showFeaturemaps) {
          for (let i = 0; i < 513; i += 10) {
            for (let j = 0; j < 513; j += 10) {
              let n = i * 513 + j;
              let max_p = 0.0;
              let max_p_k = -1;
              for (let k = 0; k < 17; k++) {
                let p = heatmap_array[17 * n + k];
                if (p > max_p) {
                  max_p = p;
                  max_p_k = k;
                }
              }
              // if (max_p > 0.3 && max_p_k >= 0) {
              //   // find offsets
              //   let dy = offsets_array[17 * (2 * n) + max_p_k];
              //   let dx = offsets_array[17 * (2 * n + 1) + max_p_k];

              //   ctx2.beginPath();
              //   ctx2.moveTo(j, i);
              //   ctx2.lineTo(j + dx, i + dy);
              //   ctx2.strokeStyle = "white";
              //   ctx2.stroke();
              // }
              if (max_p > 0.3 && max_p_k >= 0) {
                for (let m = 0; m < edges.length; m++) {
                  let p_id = edges[m][0];
                  let c_id = edges[m][1];
                  if (p_id == max_p_k) {
                    // find displacement fwd
                    let dy = displacement_fwd_array[16 * (2 * n) + m];
                    let dx = displacement_fwd_array[16 * (2 * n + 1) + m];
                    ctx2.beginPath();
                    ctx2.moveTo(j, i);
                    ctx2.lineTo(j + dx, i + dy);
                    ctx2.strokeStyle = "white";
                    ctx2.stroke();
                  }
                }

                for (let m = 0; m < edges_bwd.length; m++) {
                  let p_id = edges_bwd[m][0];
                  let c_id = edges_bwd[m][1];
                  if (p_id == max_p_k) {
                    // find displacement bwd
                    let dy = displacement_bwd_array[16 * (2 * n) + m];
                    let dx = displacement_bwd_array[16 * (2 * n + 1) + m];
                    ctx2.beginPath();
                    ctx2.moveTo(j, i);
                    ctx2.lineTo(j + dx, i + dy);
                    ctx2.strokeStyle = "green";
                    ctx2.stroke();
                  }
                }
              }
            }
          }
        }
        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }



    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    poses.forEach(({score, keypoints}) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          // drawKeypoints(keypoints, minPartConfidence, ctx);
          drawKeypoints(keypoints, minPartConfidence, ctx2);
        }
        if (guiState.output.showSkeleton) {
          // drawSkeleton(keypoints, minPartConfidence, ctx);
          drawSkeleton(keypoints, minPartConfidence, ctx2);
        }
        if (guiState.output.showBoundingBox) {
          // drawBoundingBox(keypoints, ctx);
          drawBoundingBox(keypoints, minPartConfidence, ctx2);
        }
      }
    });

    // End monitoring code for frames per second
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off the detectPoseInRealTime function.
 */
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
        'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
bindPage();
