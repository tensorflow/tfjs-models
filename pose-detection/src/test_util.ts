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

import {PoseDetectorInput} from './types';

/** Karma server directory serving local files. */
export const KARMA_SERVER = './base/src/test_data';

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

export async function loadVideo(
    videoPath: string, fps: number,
    callback: (video: PoseDetectorInput, timestamp: number) => Promise<void>) {
  // We override video's timestamp with a fake timestamp.
  let simulatedTimestamp: number;

  const interval = 1 / fps;

  // Create a video element on the html page and serve the content through karma
  const video = document.createElement('video');
  const source = document.createElement('source');
  source.src = `${KARMA_SERVER}/${videoPath}`;
  source.type = 'video/mp4';
  video.appendChild(source);
  document.body.appendChild(video);

  const promise = new Promise((resolve, reject) => {
    video.onseeked = async () => {
      await callback(video, simulatedTimestamp);
      const nextTime = video.currentTime + interval;
      if (nextTime < video.duration) {
        video.currentTime = nextTime;
        // We set the timestamp increment to 33.333 microseconds to simulate
        // the 30 fps video input. We do this so that the filter uses the
        // same fps as the python test.
        // https://github.com/google/mediapipe/blob/ecb5b5f44ab23ea620ef97a479407c699e424aa7/mediapipe/python/solution_base.py#L297
        simulatedTimestamp += 33.333;
      } else {
        resolve();
      }
    };
  });

  video.onloadedmetadata = () => {
    video.currentTime = 0;
    simulatedTimestamp = 0;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    // Must set below two lines, otherwise video width and height are 0.
    video.width = videoWidth;
    video.height = videoHeight;
  };

  return promise;
}
