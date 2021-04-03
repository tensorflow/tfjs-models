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
import * as posedetection from '@tensorflow-models/pose-detection';

import * as params from './params';
import {isMobile} from './util';

export class Camera {
  constructor() {
    this.video = document.getElementById('video');
    this.canvas = document.getElementById('output');
    this.ctx = this.canvas.getContext('2d');
  }

  /**
   * Initiate a Camera instance and wait for the camera stream to be ready.
   * @param cameraParam From app `STATE.camera`.
   */
  static async setupCamera(cameraParam) {
    const {targetFPS, sizeOption} = cameraParam;
    const $size = params.VIDEO_SIZE[sizeOption];
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

  /**
   * Draw the keypoints and skeleton on the video.
   * @param pose A pose with keypoints to render.
   * @param shouldScale If the keypoints are normalized, shouldScale should be
   *     set to true.
   */
  drawResult(pose, shouldScale = false) {
    if (pose.keypoints != null) {
      const scaleX = shouldScale ? this.video.videoWidth : 1;
      const scaleY = shouldScale ? this.video.videoHeight : 1;

      this.drawKeypoints(pose.keypoints, scaleY, scaleX);
      this.drawSkeleton(pose.keypoints, scaleY, scaleX);
    }
  }

  /**
   * Draw the keypoints on the video.
   * @param keypoints A list of keypoints, may be normalized.
   * @param scaleY If keypoints are normalized, y needs to be scaled back based
   *     on the scaleY.
   * @param scaleX If keypoints are normalized, x needs to be scaled back based
   *     on the scaleX..
   */
  drawKeypoints(keypoints, scaleY, scaleX) {
    const keypointInd =
        posedetection.util.getKeypointIndexBySide(params.STATE.model.model);
    this.ctx.fillStyle = 'White';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    keypointInd.middle.forEach(
        i => this.drawKeypoint(keypoints[i], scaleY, scaleX));

    this.ctx.fillStyle = 'Green';
    keypointInd.left.forEach(
        i => this.drawKeypoint(keypoints[i], scaleY, scaleX));

    this.ctx.fillStyle = 'Orange';
    keypointInd.right.forEach(
        i => this.drawKeypoint(keypoints[i], scaleY, scaleX));
  }

  drawKeypoint(keypoint, scaleY, scaleX) {
    // If score is null, just show the keypoint.
    const score = keypoint.score != null ? keypoint.score : 1;
    const scoreThreshold =
        params.STATE.model[params.STATE.model.model].scoreThreshold || 0;

    if (score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(
          keypoint.x * scaleX, keypoint.y * scaleY, params.DEFAULT_RADIUS, 0,
          2 * Math.PI);
      this.ctx.fill(circle);
      this.ctx.stroke(circle);
    }
  }

  /**
   * Draw the skeleton of a body on the video.
   * @param keypoints A list of keypoints, may be normalized.
   * @param scaleY If keypoints are normalized, y needs to be scaled back based
   *     on the scaleY.
   * @param scaleX If keypoints are normalized, x needs to be scaled back based
   *     on the scaleX..
   */
  drawSkeleton(keypoints, scaleY, scaleX) {
    this.ctx.fillStyle = 'White';
    this.ctx.strokeStyle = 'White';
    this.ctx.lineWidth = params.DEFAULT_LINE_WIDTH;

    posedetection.util.getAdjacentPairs(params.STATE.model.model)
        .forEach(([i, j]) => {
          const kp1 = keypoints[i];
          const kp2 = keypoints[j];

          // If score is null, just show the keypoint.
          const score1 = kp1.score != null ? kp1.score : 1;
          const score2 = kp2.score != null ? kp2.score : 1;
          const scoreThreshold =
              params.STATE.model[params.STATE.model.model].scoreThreshold || 0;

          if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
            this.ctx.beginPath();
            this.ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
            this.ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
            this.ctx.stroke();
          }
        });
  }
}
