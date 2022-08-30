/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
let video, model, stats;
let estimator;
let faceDetector;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let applyMask;
let canvasEl;
let gl;
let inputTextureFrameBuffer;

const statusEl = document.querySelector('#status');

function createCanvas() {
  canvasEl = document.createElement('canvas');
  mouseX = videoWidth / 2;
  mouseY = videoHeight / 2;
  canvasEl.width = videoWidth;
  canvasEl.height = videoHeight;
  canvasEl.style.width = `${videoWidth}px`;
  canvasEl.style.height = `${videoHeight}px`;
  canvasEl.addEventListener('mousedown', ev => {
    const rect = ev.target.getBoundingClientRect();
    // The user-facing camera is mirrored, flip horizontally.
    mouseX = rect.right - ev.clientX + rect.left;
    mouseY = ev.clientY - rect.top;
  }, false);

  const wrapper = document.querySelector('.main');
  wrapper.innerHTML = '';
  wrapper.appendChild(canvasEl);

  gl = getWebGLRenderingContext(canvasEl);
  applyMask = new MaskStep(gl);
  inputTextureFrameBuffer =
      createTextureFrameBuffer(gl, gl.LINEAR, videoWidth, videoHeight);
}

/**
 * Returns a pair of transform from an interval to another interval.
 * @param {number} fromMin - min of the start interval.
 * @param {number} fromMax - max of the start interval.
 * @param {number} toMin - min of the ending interval.
 * @param {number} toMax - max of the ending interval.
 */
function transformValueRange(fromMin, fromMax, toMin, toMax) {
  const fromRange = fromMax - fromMin;
  const ToRange = toMax - toMin;
  const scale = ToRange / fromRange;
  const offset = toMin - fromMin * scale;
  return {scale, offset};
}

async function init() {
  createCanvas();

  const customBackendName = 'custom-webgl';

  const kernels = tf.getKernelsForBackend('webgl');
  kernels.forEach(kernelConfig => {
    const newKernelConfig = {...kernelConfig, backendName: customBackendName};
    tf.registerKernel(newKernelConfig);
  });
  tf.registerBackend(customBackendName, () => {
    return new tf.MathBackendWebGL(new tf.GPGPUContext(gl));
  });
  await tf.setBackend(customBackendName);

  let cachedData = null;
  const predict = async () => {
    // Put original video content on the input texture.
    inputTextureFrameBuffer.bindTexture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, video);

    let data;
    if (numInferences % STATE.DepthCachedFrames === 0) {
      if (cachedData) {
        // Make sure to dispose all tensors, otherwise there will be memory
        // leak.
        cachedData.tensorRef.dispose();
      }

      // Extract face.
      const faces = await faceDetector.estimateFaces(video);

      let output4D;
      if (faces.length === 0) {
        output4D = tf.zeros([videoHeight, videoWidth, 1]);
      } else {
        const face = faces[0];

        const rightEye = face.keypoints[0];
        const leftEye = face.keypoints[1];
        const nose = face.keypoints[2];

        const rightEyePoint = [rightEye.x, rightEye.y];
        const leftEyePoint = [leftEye.x, leftEye.y];
        const eyesDistance =
            Math.sqrt(tf.util.distSquared(rightEyePoint, leftEyePoint));

        const scaleDistTo64 = 64 / eyesDistance;

        let portraitHeight, portraitWidth;
        let sliceStart, sliceSize;
        let rescaledEyeHeight, rescaledEyeWidth;
        const portraitPadded = tf.tidy(() => {
          const image1Tensor = tf.browser.fromPixels(video);
          const image1Float = tf.cast(image1Tensor, 'float32');
          // Resize until distance of eyes is 192 / 3 = 94.
          const rescaledEye = tf.image.resizeBilinear(image1Float, [
            Math.round(videoHeight * scaleDistTo64),
            Math.round(videoWidth * scaleDistTo64)
          ]);
          [rescaledEyeHeight, rescaledEyeWidth] = rescaledEye.shape;

          // Find position of nose in rescaled image.
          const nosePoint = [
            Math.round(nose.x * scaleDistTo64),
            Math.round(nose.y * scaleDistTo64)
          ];

          sliceStart = [
            Math.max(0, nosePoint[1] - 256 / 2),
            Math.max(0, nosePoint[0] - 192 / 2), 0
          ];
          sliceSize = [
            Math.min(rescaledEyeHeight - sliceStart[0], 256),
            Math.min(rescaledEyeWidth - sliceStart[1], 192), 3
          ];

          // Make a crop centered at the nose.
          const portraitCrop = tf.slice(rescaledEye, sliceStart, sliceSize);
          [portraitHeight, portraitWidth] = sliceSize;

          // Pad if the not enough pixels are around the nose.
          return tf.pad(
              portraitCrop,
              [[0, 256 - portraitHeight], [0, 192 - portraitWidth], [0, 0]]);
        });

        beginEstimateSegmentationStats();

        const output = await estimator.estimateDepth(
            portraitPadded,
            {minDepth: STATE.MinDepth, maxDepth: STATE.MaxDepth});

        portraitPadded.dispose();

        const depthMap = await output.toTensor();

        tf.tidy(() => {
          // Remove padding added in `portraitPadded`.
          const outputUnpadded =
              tf.slice(depthMap, 0, [portraitHeight, portraitWidth]);
          depthMap.dispose();

          // Add pixels removed by nose crop in `portraotCrop`.
          const outputRescaledEye = tf.pad(outputUnpadded, [
            [sliceStart[0], rescaledEyeHeight - (sliceStart[0] + sliceSize[0])],
            [sliceStart[1], rescaledEyeWidth - (sliceStart[1] + sliceSize[1])],
          ]);

          const outputChannel =
              tf.expandDims(tf.expandDims(outputRescaledEye, 0), 3);

          // Rescale to original input size.
          const outputResized =
              tf.image.resizeBilinear(outputChannel, [videoHeight, videoWidth]);

          // Pad from 1 channel to 4 channel to match input image channels.
          output4D = tf.pad(outputResized, [[0, 0], [0, 0], [0, 0], [0, 3]], 1);

          endEstimateSegmentationStats();

          return output4D;
        });
      }
      try {
        // Get the tensor result and the texture that holds the data.
        // We tell the system to use the video width and height as the tex
        // shape, this allows the densely packed data to have the same layout
        // as the original video content, which simplifies the shader logic.
        // This only works if the data shape is [1, height, width, 4].
        data = output4D.dataToGPU({customTexShape: [videoHeight, videoWidth]});
        // If data happens to be on CPU.
      } catch (error) {
        output4D.dispose();
        requestAnimationFrame(predict);
        return;
      }
      // Sync data to CPU to ensure sync and FPS is accurate.
      output4D.dataSync();

      // Ensure GPU is done for timing purposes.
      const webGLBackend = tf.backend();
      const buffer =
          webGLBackend.gpgpu.createBufferFromTexture(data.texture, 1, 1);
      webGLBackend.gpgpu.downloadFloat32MatrixFromBuffer(buffer, 1);
    } else {
      data = cachedData;
    }

    // Combine the input texture and tensor texture with additional shader
    // logic. In this case, we just pass through foreground pixels and make
    // background pixels more transparent.
    const result = applyMask.process(
        inputTextureFrameBuffer,
        createTexture(gl, data.texture, videoWidth, videoHeight));

    // Making gl.DRAW_FRAMEBUFFER to be null sets rendering back to default
    // framebuffer.
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    // Caching the data of the result texture to be drawn in the
    // gl.READ_FRAMEBUFFER.
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, result.framebuffer_);
    // Transfer the data from read framebuffer to the default framebuffer to
    // make it show on canvas.
    gl.blitFramebuffer(
        0, 0, videoWidth, videoHeight, 0, videoHeight, videoWidth, 0,
        gl.COLOR_BUFFER_BIT, gl.LINEAR);

    // Make sure to dispose all tensors, otherwise there will be memory leak.
    // data.tensorRef.dispose();
    cachedData = data;

    requestAnimationFrame(predict);
  };

  predict();
}

setupPage();
