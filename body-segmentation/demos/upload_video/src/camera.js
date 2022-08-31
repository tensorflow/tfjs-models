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
 import * as params from './shared/params';

 export class Context {
   constructor() {
     this.video = document.getElementById('video');
     this.canvas = document.getElementById('output');
     this.source = document.getElementById('currentVID');
     this.ctx = this.canvas.getContext('2d');
     const stream = this.canvas.captureStream();
     const options = {mimeType: 'video/webm; codecs=vp9'};
     this.mediaRecorder = new MediaRecorder(stream, options);
     this.mediaRecorder.ondataavailable = this.handleDataAvailable;
   }

   drawToCanvas(canvas) {
     this.ctx.drawImage(canvas, 0, 0, this.video.videoWidth, this.video.videoHeight);
   }

   drawFromVideo(ctx) {
     ctx.drawImage(
         this.video, 0, 0, this.video.videoWidth, this.video.videoHeight);
   }

   clearCtx() {
     this.ctx.clearRect(0, 0, this.video.videoWidth, this.video.videoHeight);
   }

   start() {
     this.mediaRecorder.start();
   }

   stop() {
     this.mediaRecorder.stop();
   }

   handleDataAvailable(event) {
     if (event.data.size > 0) {
       const recordedChunks = [event.data];

       // Download.
       const blob = new Blob(recordedChunks, {type: 'video/webm'});
       const url = URL.createObjectURL(blob);
       const a = document.createElement('a');
       document.body.appendChild(a);
       a.style = 'display: none';
       a.href = url;
       a.download = 'body-segmentation.webm';
       a.click();
       window.URL.revokeObjectURL(url);
     }
   }
 }
