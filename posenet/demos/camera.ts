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
import {GUI} from 'dat.gui';

import * as posenet from '../src';
import {OutputStride} from '../src/posenet';

// tslint:disable-next-line:max-line-length
import {drawKeypoints, drawSkeleton, renderToCanvas} from './demo_util';

const maxStride = 32;

const videoSizes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map(
    (multiplier: number) => (maxStride * multiplier + 1));

const maxVideoSize = 513;

async function getCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();

  return devices.filter(({kind}) => kind === 'videoinput');
}

let currentStream: MediaStream = null;

function stopCurrentVideoStream() {
  if (currentStream) {
    currentStream.getTracks().forEach(track => {
      track.stop();
    });
  }
}

function loadVideo(cameraId: string) {
  return new Promise<HTMLVideoElement>((resolve, reject) => {
    stopCurrentVideoStream();
    const video = document.getElementById('video') as HTMLVideoElement;
    video.width = maxVideoSize;
    video.height = maxVideoSize;

    if (navigator.getUserMedia) {
      navigator.getUserMedia(
          {
            video: {
              width: maxVideoSize,
              height: maxVideoSize,
              deviceId: {exact: cameraId}
            }
          },
          handleVideo, videoError);
    }

    function handleVideo(stream: MediaStream) {
      currentStream = stream;
      video.src = window.URL.createObjectURL(stream);

      resolve(video);
    }

    function videoError(e: MediaStreamError) {
      // do something
      reject(e);
    }
  });
}

type GuiState = {
  camera?: string,
        outputStride: OutputStride,
        minPartConfidence: number,
        minPoseConfidence: number,
        maxPoseDetections: number,
        nmsRadius: number,
        videoResolution: number,
        showVideo: boolean,
        showSkeleton: boolean,
        showPoints: boolean,
        outputResolution: number
};

const guiState: GuiState = {
  outputStride: 16,
  minPartConfidence: 0.2,
  minPoseConfidence: 0.4,
  videoResolution: 225,
  maxPoseDetections: 2,
  nmsRadius: 10,
  showVideo: true,
  showSkeleton: true,
  showPoints: true,
  outputResolution: 385
};

function setupGui(cameras: MediaDeviceInfo[]) {
  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const cameraOptions = cameras.reduce(
      (result: {[label: string]: string}, {label, deviceId}):
          {[label: string]: string} => {
            result[label] = deviceId;
            return result;
          },
      {});

  const gui = new GUI();
  gui.add(guiState, 'camera', cameraOptions).onChange((deviceId: string) => {
    loadVideo(deviceId);
  });
  gui.add(guiState, 'outputStride', [8, 16, 32]);
  gui.add(guiState, 'minPartConfidence', 0.0, 1.0);
  gui.add(guiState, 'minPoseConfidence', 0.0, 1.0);
  gui.add(guiState, 'maxPoseDetections').min(1).max(20).step(1);
  gui.add(guiState, 'nmsRadius').min(0.0).max(40.0);
  gui.add(guiState, 'videoResolution', videoSizes);
  gui.add(guiState, 'outputResolution').min(0).max(800).step(1);
  gui.add(guiState, 'showVideo');
  gui.add(guiState, 'showSkeleton');
  gui.add(guiState, 'showPoints');
}

function detectPoseInRealTime(video: HTMLVideoElement, model: posenet.PoseNet) {
  const canvas = document.getElementById('output') as HTMLCanvasElement;
  const ctx = canvas.getContext('2d');

  async function poseDetectionFrame() {
    const start = new Date().getTime();
    const videoResolution = Number(guiState.videoResolution);
    const outputStride = Number(guiState.outputStride) as 8 | 16 | 32;
    const minConfidence = Number(guiState.minPartConfidence);
    const outputResolution = Number(guiState.outputResolution);
    canvas.width = outputResolution;
    canvas.height = outputResolution;

    ctx.clearRect(0, 0, outputResolution, outputResolution);
    ctx.translate(outputResolution, 0);
    ctx.scale(-1, 1);

    const originalImage = tf.fromPixels(video);
    const image =
        originalImage.resizeBilinear([videoResolution, videoResolution]);

    const scale = outputResolution / videoResolution;

    const startPredict = new Date().getTime();

    const poses = await model.predictAndDecodeMultiplePoses(
        image,
        outputStride,
        guiState.maxPoseDetections,
        guiState.minPartConfidence,
        guiState.nmsRadius,
    );

    console.log(
        'total prediction and decode time',
        new Date().getTime() - startPredict);

    if (guiState.showVideo) {
      const toRender = await tf.tidy(
          () => originalImage.reverse(1).resizeBilinear(
              [outputResolution, outputResolution]));
      await renderToCanvas(toRender, ctx);

      toRender.dispose();
    }

    poses.forEach(({score, keypoints}) => {
      if (score >= guiState.minPoseConfidence) {
        if (guiState.showPoints) {
          drawKeypoints(keypoints, minConfidence, ctx, scale);
        }
        if (guiState.showSkeleton) {
          drawSkeleton(keypoints, minConfidence, ctx, scale);
        }
      }
    });

    console.log('after render models in memory', tf.memory().numTensors);

    image.dispose();
    originalImage.dispose();
    console.log('frame time', new Date().getTime() - start);
    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

export async function bindPage() {
  const model = new posenet.PoseNet();

  await model.load();

  document.getElementById('loading').setAttribute('style', 'display:none');
  document.getElementById('main').setAttribute('style', 'display:block');

  const cameras = await getCameras();
  if (cameras.length === 0) {
    alert('No webcams available.  Reload the page when a webcam is available.');
    return;
  }

  const video = await loadVideo(cameras[0].deviceId);
  setupGui(cameras);

  detectPoseInRealTime(video, model);
}
