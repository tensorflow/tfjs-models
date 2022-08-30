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
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let applyMask;
let canvasEl;
let gl;
let inputTextureFrameBuffer;
let mouseX, mouseY;

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
    beginEstimateSegmentationStats();

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
      const depthTensor =
          await estimator
              .estimateDepth(
                  video, {minDepth: STATE.MinDepth, maxDepth: STATE.MaxDepth})
              .then(depthMap => depthMap.toTensor());

      tf.tidy(() => {
        const depth4D =
            tf.reshape(depthTensor, [1, videoHeight, videoWidth, 1]);
        depthTensor.dispose();
        // Pad from 1 channel to 4 channel to match input image channels.
        const depth4DPadded =
            tf.pad(depth4D, [[0, 0], [0, 0], [0, 0], [0, 3]], 1);

        // Get the tensor result and the texture that holds the data.
        // We tell the system to use the video width and height as the tex
        // shape, this allows the densely packed data to have the same layout
        // as the original video content, which simplifies the shader logic.
        // This only works if the data shape is [1, height, width, 4].
        data = depth4DPadded.dataToGPU(
            {customTexShape: [videoHeight, videoWidth]});
        // Sync data to CPU to ensure sync and FPS is accurate.
        depth4DPadded.dataSync();
        return data.tensorRef;
      });
    } else {
      data = cachedData;
    }

    // Combine the input texture and tensor texture with additional shader
    // logic. In this case, we just pass through foreground pixels and make
    // background pixels more transparent.
    const result = applyMask.process(
        inputTextureFrameBuffer,
        createTexture(gl, data.texture, videoWidth, videoHeight),
        [mouseX, mouseY]);

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
    endEstimateSegmentationStats();

    requestAnimationFrame(predict);
  };

  predict();
}

setupPage();
