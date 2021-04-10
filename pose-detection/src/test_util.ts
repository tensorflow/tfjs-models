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
    videoPath: string, callback: (video: HTMLVideoElement) => Promise<void>) {
  const video = document.createElement('video');
  const source = document.createElement('source');
  source.src = `${KARMA_SERVER}/${videoPath}`;
  source.type = 'video/mp4';
  video.appendChild(source);
  document.body.appendChild(video);
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  const promise = new Promise((resolve, reject) => {
    video.onseeked = async () => {
      canvas.height = video.videoHeight;
      canvas.width = video.videoWidth;

      ctx.drawImage(video, 200, 0);
      await callback(video);
      const nextTime = video.currentTime + 0.2;
      if (nextTime < video.duration) {
        video.currentTime = nextTime;
      } else {
        resolve();
      }
    };
  });

  video.onloadedmetadata = () => {
    video.currentTime = 0.001;
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;
    // Must set below two lines, otherwise video width and height are 0.
    video.width = videoWidth;
    video.height = videoHeight;
  };

  return promise;
}
