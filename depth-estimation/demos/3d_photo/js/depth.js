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
let segmentationModel;
let segmenter;
let estimationModel;
let estimator;
let resultCanvas;
let image1;
let mask1;
let perf;
let masked;

const src_images = [image1, mask1];
const WIDTH = 192;
const HEIGHT = 256;
const IMAGE_PREFIX = 'im';

/**
 * Changes the input image.
 * @param {!Object} file - the input image file.
 * @param {string} image_id - the input image id.
 */
function changeImage(file, image_id) {
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = document.createElement('img');
    img.src = e.target.result

    const imgContainer = document.createElement('div');
    imgContainer.classList.add('img-container');
    imgContainer.appendChild(img);

    const uploadedImg = document.getElementById('uploaded-img');
    // clean result before
    uploadedImg.innerHTML = '';
    // append new image
    uploadedImg.appendChild(imgContainer);

    new Cropper(img, {
      cropBoxResizable: true,
      aspectRatio: WIDTH / HEIGHT,
      guides: false,
      movable: false,
      rotatable: false,
      scalable: false,
      zoomable: false,
      zoomOnTouch: false,
      zoomOnWheel: false,
      viewMode: 1,
      dragMode: 'none',
      crop(event) {
        const resizeCanvas = document.getElementById('resize');
        const context = resizeCanvas.getContext('2d');
        context.clearRect(0, 0, WIDTH, HEIGHT);
        context.drawImage(
            img, event.detail.x, event.detail.y, event.detail.width,
            event.detail.height, 0, 0, WIDTH, HEIGHT);

        const changeImg = document.getElementById(image_id);
        changeImg.src = resizeCanvas.toDataURL();
        changeImg.width = WIDTH;
        changeImg.height = HEIGHT;
        changeImg.onload = function() {
          predict();
        }
      },
    });
  };
  reader.readAsDataURL(file);
}

/**
 * Loads an image to the HTML image element.
 * @param {string} filename - the input image file.
 * @param {string} element_id - the target element id.
 */
function loadImage(filename, element_id) {
  let _img = document.getElementById(element_id);
  let newImg = new Image;
  newImg.onload = function() {
    _img.src = this.src;
    predict();
  };

  newImg.src = 'images/' + filename;
}


/**
 * Loads an image to the HTML canvas element.
 * @param {string} filename - the input image file.
 * @param {string} element_id - the target element id.
 */
function loadImageToCanvas(filename, element_id) {
  let _mask = document.getElementById(element_id).getContext('2d');
  let _img = new Image;
  _img.onload = function() {
    _mask.drawImage(_img, 0, 0, _img.width, _img.height, 0, 0, WIDTH, HEIGHT);
  };

  _img.src = 'images/' + filename;
}

/**
 * Loads the preset.
 * @param {number} preset_id - preset id.
 */
function loadPreset(preset_id) {
  const uploadedImg = document.getElementById('uploaded-img');
  // clean result before
  uploadedImg.innerHTML = '';

  const deleteUploadButton = document.getElementById('delete-upload');
  if (deleteUploadButton.style.visibility === 'visible') {
    deleteUploadButton.click();
  }

  loadImage(IMAGE_PREFIX + preset_id + '.jpg', 'im1');
}

/**
 * Runs the model.
 */
function predict() {
  // Tests if the model is loaded.
  if (segmentationModel == null || segmenter == null ||
      estimationModel == null || estimator == null) {
    alert('Model is not available!');
    return;
  }

  // Tests if an image is missing.
  for (let src_image in src_images) {
    if (src_image.height === 0 || src_image.width === 0) {
      alert('You need to upload an image!');
      return;
    }
  }

  capturer = null;
  capturerInitialTheta = null;

  predictButton.textContent = 'Running...';
  predictButton.disabled = true;

  // Sets timeout = 0 to force reload the UI.
  setTimeout(function() {
    const start = Date.now();
    const ctx = resultCanvas.getContext('2d');
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);

    const getPortraitDepth = async () => {
      const segmentation = await segmenter.segmentPeople(image1);

      // Convert the segmentation into a mask to darken the background.
      const foregroundColor = {r: 0, g: 0, b: 0, a: 0};
      const backgroundColor = {r: 0, g: 0, b: 0, a: 255};
      const backgroundDarkeningMask = await bodySegmentation.toBinaryMask(
          segmentation, foregroundColor, backgroundColor);
      segmentation.map(
          singleSegmentation => singleSegmentation.mask.toTensor().then(
              tensor => tensor.dispose()));

      const opacity = 1.0;
      const maskBlurAmount = 0;
      const flipHorizontal = false;

      // Draw the mask onto the image on a canvas.  With opacity set to 0.7
      // and maskBlurAmount set to 3, this will darken the background and blur
      // the darkened background's edge.
      await bodySegmentation.drawMask(
          masked, image1, backgroundDarkeningMask, opacity, maskBlurAmount,
          flipHorizontal);

      const result = await estimator.estimateDepth(
          image1, {minDepth: config.minDepth, maxDepth: config.maxDepth});
      const depthMap = await result.toTensor();

      tf.tidy(() => {
        const depthMap3D = tf.expandDims(depthMap, axis = 2);
        const transformNormalize =
            transformValueRange(0, 1, 0, 255 * 255 * 255);
        let depth_rescale = tf.add(
            tf.mul(depthMap3D, transformNormalize.scale),
            transformNormalize.offset);

        let depth_r = tf.floorDiv(depth_rescale, 255.0 * 255.0);
        let depth_remain =
            tf.floorDiv(tf.mod(depth_rescale, 255.0 * 255.0), 1.0);
        let depth_g = tf.floorDiv(depth_remain, 255);
        let depth_b = tf.floorDiv(tf.mod(depth_remain, 255), 1.0);

        let depth_rgb = tf.concat([depth_r, depth_g, depth_b], axis = 2);

        // Renders the result on a canvas.
        const transformBack = transformValueRange(0, 255, 0, 1);

        // Converts back to 0-1.
        const rgbFinal = tf.clipByValue(
            tf.add(
                tf.mul(depth_rgb, transformBack.scale), transformBack.offset),
            0, 1);

        tf.browser.toPixels(rgbFinal, resultCanvas);
      });

      depthMap.dispose();

      const end = Date.now();
      const time = end - start;
      perf.textContent = `E2E latency: ${time}ms`;
      predictButton.textContent = 'Measure Latency';
      predictButton.disabled = false;

      setTimeout(() => {
        updateDepthCallback();
        canvas_texture.needsUpdate = true;
      }, 500);
    };
    getPortraitDepth();
  }, 0);
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

function isMobile() {
  return /Mobile|Android|iP(hone|od)|IEMobile|BlackBerry|Kindle|Silk-Accelerated|(hpw|web)OS|Opera M(obi|ini)/
      .test(navigator.userAgent);
}
/**
 * Sets up the page.
 */
async function setupPage() {
  predictButton = document.getElementById('predict');
  resultCanvas = document.getElementById('result');
  image1 = document.getElementById('im1');
  masked = document.getElementById('masked');

  perf = document.getElementById('perf');

  if (isMobile()) {
    const elements = document.getElementsByClassName('desktop');
    elements.forEach(element => element.style.visibility = 'hidden');
  }

  try {
    segmentationModel =
        bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
  } catch (e) {
    predictButton.textContent = 'Error in loading segmentation model.';
  }
  segmenter = await bodySegmentation.createSegmenter(
      segmentationModel, {runtime: 'tfjs'});

  try {
    estimationModel = depthEstimation.SupportedModels.ARPortraitDepth;
  } catch (e) {
    predictButton.textContent = 'Error in loading estimation model.';
  }
  estimator = await depthEstimation.createEstimator(estimationModel);
  predict();

  predictButton.textContent = 'Measure Latency';
  predictButton.disabled = false;

  // Set up the upload image area.
  const uploadedImage = document.getElementById('uploaded-img');
  const deleteUploadButton = document.getElementById('delete-upload');
  const rightSide = document.getElementById('right-side');
  const dropzoneForm = document.getElementById('dropzone');
  let dropzone;
  new Dropzone('#dropzone', {
    transformFile: function(file, done) {
      dropzone = this;

      rightSide.removeChild(dropzoneForm);
      changeImage(file, 'im1');
      setTimeout(() => {
        deleteUploadButton.style.visibility = 'visible';
      }, 500);
    }
  });

  // Allow removal of the uploaded image.
  deleteUploadButton.addEventListener('click', function() {
    uploadedImage.innerHTML = '';
    rightSide.insertBefore(dropzoneForm, uploadedImage);
    dropzone.removeAllFiles(true);
    deleteUploadButton.style.visibility = 'hidden';
  });

  initGL();
  animate();
}

setupPage();
