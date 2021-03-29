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
import * as posedetection from '@tensorflow-models/posedetection';

import {VIDEO_SIZE} from './params';
import {isMobile, sigmoid} from './util';

export class Camera {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output');
    this.ctx = this.canvas.getContext('2d');

    // The video frame rate may be lower than the browser animate frame
    // rate. We use this to avoid processing the same frame twice.
    this.lastVideoTime = 0;
  }

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static async setupCamera(cameraParam) {
    const {targetFPS, sizeOption} = cameraParam;
    const $size = VIDEO_SIZE[sizeOption];
    const videoConfig = {
      'audio': false,
      'video': {
        facingMode: 'user',
        // Only setting the video to a specified size for large screen, on
        // mobile devices accept the default size.
        width: isMobile() ? undefined : $size.width,
        height: isMobile() ? undefined : $size.height,
        frameRate: {
          ideal: targetFPS,
        }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(videoConfig);

    const camera = new Camera();
    camera.video.srcObject = stream;

    await new Promise((resolve) => {
      camera.video.onloadedmetadata = () => {
        resolve(video);
      };
    });

    camera.video.play();

    const videoWidth = camera.video.videoWidth;
    const videoHeight = camera.video.videoHeight;
    // Must set below two lines, otherwise video element doesn't show.
    camera.video.width = videoWidth;
    camera.video.height = videoHeight;

    camera.canvas.width = videoWidth;
    camera.canvas.height = videoHeight;
    const canvasContainer = document.querySelector('.canvas-wrapper');
    canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

    // Because the image from camera is mirrored, need to flip horizontally.
    camera.ctx.translate(camera.video.videoWidth, 0);
    camera.ctx.scale(-1, 1);

    return camera;
  }

  drawCtx() {
    this.ctx.drawImage(
        this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  clearCtx() {
    this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
  }

  drawResult(pose, model, confidence) {
    this.drawKeypoints(pose.keypoints, model, confidence);
    this.drawSkeleton(pose.keypoints, model, confidence);
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints, may be normalized.
   * @param shouldScale If the keypoints are normalized, shouldScale should be
   *     set to true.
   */
  drawKeypoints(keypoints, model, confidence) {
    const shouldScale =
        model === posedetection.SupportedModels.MediapipeBlazepose;
    const scaleX = shouldScale ? this.video.videoWidth : 1;
    const scaleY = shouldScale ? this.video.videoHeight : 1;

    const keypointInd = posedetection.util.getKeypointIndexBySide(model);
    this.ctx.fillStyle = 'LightCyan';
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 2;

    keypointInd.middle.forEach(
        keypointInd => this.drawKeypoint(
            keypoints[keypointInd], scaleY, scaleX, confidence));
    this.ctx.fillStyle = 'green';
    keypointInd.left.forEach(
        keypointInd => this.drawKeypoint(
            keypoints[keypointInd], scaleY, scaleX, confidence));
    this.ctx.fillStyle = 'orange';
    keypointInd.right.forEach(
        keypointInd => this.drawKeypoint(
            keypoints[keypointInd], scaleY, scaleX, confidence));
  }

  drawKeypoint(keypoint, scaleY, scaleX, confidence) {
    let score = 1;
    let scoreP = 1;
    if (keypoint.visibility) {
      score = sigmoid(keypoint.visibility);
      scoreP = sigmoid(keypoint.score);
    }

    if (score > confidence && scoreP > confidence) {
      const circle = new Path2D();
      circle.arc(keypoint.x * scaleX, keypoint.y * scaleY, 4, 0, 2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  /**
   * Draws a pose skeleton by looking up all adjacent keypoints/joints
   */
  drawSkeleton(keypoints, model, confidence) {
    const shouldScale =
        model === posedetection.SupportedModels.MediapipeBlazepose;
    const scaleX = shouldScale ? this.video.videoWidth : 1;
    const scaleY = shouldScale ? this.video.videoHeight : 1;
    this.ctx.fillStyle = 'red';
    this.ctx.strokeStyle = 'white';
    this.ctx.lineWidth = 4;

    posedetection.util.getAdjacentKeypointIndexPair(model).forEach(
        ([keypointAInd, keypointBInd]) => {
          const keypointA = keypoints[keypointAInd];
          const keypointB = keypoints[keypointBInd];

          let scoreA = 1;
          let scoreB = 1;
          let scoreAP = 1;
          let scoreBP = 1;
          if (keypointA.visibility) {
            scoreA = sigmoid(keypointA.visibility);
            scoreB = sigmoid(keypointB.visibility);
            scoreAP = sigmoid(keypointA.score);
            scoreBP = sigmoid(keypointB.score);
          }

          if (scoreA > confidence && scoreB > confidence &&
              scoreAP > confidence && scoreBP > confidence) {
            this.ctx.beginPath();
            this.ctx.moveTo(keypointA.x * scaleX, keypointA.y * scaleY);
            this.ctx.lineTo(keypointB.x * scaleX, keypointB.y * scaleY);
            this.ctx.stroke();
          }
        });
  }
}