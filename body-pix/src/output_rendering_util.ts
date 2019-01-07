/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {cpuBlur} from './blur';
import {PartSegmentation, PersonSegmentation} from './types';

const offScreenCanvases: {[name: string]: HTMLCanvasElement} = {};

type ImageType = HTMLImageElement|HTMLVideoElement|HTMLCanvasElement;
type HasDimensions = {
  width: number,
  height: number
};

function isSafari() {
  return (/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
}

function assertSameDimensions(
    {width: widthA, height: heightA}: HasDimensions,
    {width: widthB, height: heightB}: HasDimensions, nameA: string,
    nameB: string) {
  if (widthA !== widthB || heightA !== heightB) {
    throw new Error(`error: dimensions must match. ${nameA} has dimensions ${
        widthA}x${heightA}, ${nameB} has dimensions ${widthB}x${heightB}`);
  }
}

function flipCanvasHorizontal(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
}

function drawWithCompositing(
    ctx: CanvasRenderingContext2D, image: HTMLCanvasElement|ImageType,
    compositOperation: string) {
  ctx.globalCompositeOperation = compositOperation;
  ctx.drawImage(image, 0, 0);
}

function createOffScreenCanvas(): HTMLCanvasElement {
  const offScreenCanvas = document.createElement('canvas');
  return offScreenCanvas;
}

function ensureOffscreenCanvasCreated(id: string): HTMLCanvasElement {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}

function drawAndBlurImageOnCanvas(
    image: ImageType, blurAmount: number, canvas: HTMLCanvasElement) {
  const {height, width} = image;
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  if (isSafari()) {
    cpuBlur(canvas, image, blurAmount);
  } else {
    // tslint:disable:no-any
    (ctx as any).filter = `blur(${blurAmount}px)`;
    ctx.drawImage(image, 0, 0, width, height);
  }
  ctx.restore();
}

function drawAndBlurImageOnOffScreenCanvas(
    image: ImageType, blurAmount: number,
    offscreenCanvasName: string): HTMLCanvasElement {
  const canvas = ensureOffscreenCanvasCreated(offscreenCanvasName);
  if (blurAmount === 0) {
    renderImageToCanvas(image, canvas);
  } else {
    drawAndBlurImageOnCanvas(image, blurAmount, canvas);
  }
  return canvas;
}

function renderImageToCanvas(image: ImageType, canvas: HTMLCanvasElement) {
  const {width, height} = image;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0, width, height);
}
/**
 * Draw an image on a canvas
 */
function renderImageDataToCanvas(image: ImageData, canvas: HTMLCanvasElement) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(image, 0, 0);
}

function renderImageDataToOffScreenCanvas(
    image: ImageData, canvasName: string): HTMLCanvasElement {
  const canvas = ensureOffscreenCanvasCreated(canvasName);
  renderImageDataToCanvas(image, canvas);

  return canvas;
}

/**
 * Given the output from estimating person segmentation, generates a black image
 * with opacity and transparency at each pixel determined by the corresponding
 * binary segmentation value at the pixel from the output.  In other words,
 * pixels where there is a person will be opaque and where there is not a person
 * will be transparent, and visa-versa when 'invertMask' is set to true. This
 * can be used as a mask to crop a person or the background when compositing.
 *
 * @param segmentation The output from estimagePersonSegmentation; an object
 * containing a width, height, and a binary array with 1 for the pixels that are
 * part of the person, and 0 otherwise.
 *
 * @param invertMask If the mask should be inverted. Defaults to false.  When
 * set to true, pixels where there is a person are transparent and where there
 * is no person become opaque.
 *
 * @returns An ImageData with the same width and height of the
 * personSegmentation, with opacity and transparency at each pixel determined by
 * the corresponding binary segmentation value at the pixel from the output.
 */
export function toMaskImageData(
    segmentation: PersonSegmentation, invertMask = false): ImageData {
  const {width, height, data} = segmentation;
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentation mask.
    const shouldMask = invertMask ? 1 - data[i] : data[i];
    // alpha will determine how dark the mask should be.
    const alpha = shouldMask * 255;

    const j = i * 4;
    bytes[j + 0] = 0;
    bytes[j + 1] = 0;
    bytes[j + 2] = 0;
    bytes[j + 3] = Math.round(alpha);
  }

  return new ImageData(bytes, width, height);
}

/**
 * Given the output from estimating part segmentation, and an array of colors
 * indexed by part id, generates an image with the corresponding color for each
 * part at each pixel, and black pixels where there is no part.
 *
 * @param partSegmentation   The output from estimatePartSegmentation; an object
 * containing a width, height, and an array with a part id from 0-24 for the
 * pixels that are part of a corresponding body part, and -1 otherwise.
 *
 * @param partColors A multi-dimensional array of rgb colors indexed by
 * part id.  Must have 24 colors, one for every part.
 *
 * @returns An ImageData with the same width and height of the partSegmentation,
 * with the corresponding color for each part at each pixel, and black pixels
 * where there is no part.
 */
export function toColoredPartImageData(
    partSegmentation: PartSegmentation,
    partColors: Array<[number, number, number]>): ImageData {
  const {width, height, data} = partSegmentation;
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const partId = Math.round(data[i]);
    const j = i * 4;

    if (partId === -1) {
      bytes[j + 0] = 0;
      bytes[j + 1] = 0;
      bytes[j + 2] = 0;
      bytes[j + 3] = 255;
    } else {
      const color = partColors[partId];

      if (!color) {
        throw new Error(`No color could be found for part id ${partId}`);
      }
      bytes[j + 0] = color[0];
      bytes[j + 1] = color[1];
      bytes[j + 2] = color[2];
      bytes[j + 3] = 255;
    }
  }

  return new ImageData(bytes, width, height);
}

const CANVAS_NAMES = {
  blurred: 'blurred',
  blurredMask: 'blurred-mask',
  mask: 'mask'
};

/**
 * Given an image and a maskImage of type ImageData, draws the image with the
 * mask on top of it onto a canvas.
 *
 * @param canvas The canvas to be drawn onto.
 *
 * @param image The original image to apply the mask to.
 *
 * @param maskImage An ImageData containing the mask.  Ideally this should be
 * generated by toMaskImageData or toColoredPartImageData.
 *
 * @param maskOpacity The opacity of the mask when drawing it on top of the
 * image. Defaults to 0.7. Should be a float between 0 and 1.
 *
 * @param maskBlurAmount How many pixels to blur the mask by. Defaults to 0.
 * Should be an integer between 0 and 20.
 *
 * @param flipHorizontal If the result should be flipped horizontally.  Defaults
 * to false.
 */
export function drawMask(
    canvas: HTMLCanvasElement, image: ImageType, maskImage: ImageData,
    maskOpacity = 0.7, maskBlurAmount = 0, flipHorizontal = false) {
  assertSameDimensions(image, maskImage, 'image', 'mask');

  const mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);
  const blurredMask = drawAndBlurImageOnOffScreenCanvas(
      mask, maskBlurAmount, CANVAS_NAMES.blurredMask);

  canvas.width = blurredMask.width;
  canvas.height = blurredMask.height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }

  ctx.drawImage(image, 0, 0);
  ctx.globalAlpha = maskOpacity;
  ctx.drawImage(blurredMask, 0, 0);
  ctx.restore();
}

function createBackgroundMask(
    segmentation: PersonSegmentation,
    edgeBlurAmount: number): HTMLCanvasElement {
  const invertMask = false;
  const backgroundMaskImage = toMaskImageData(segmentation, invertMask);

  const backgroundMask =
      renderImageDataToOffScreenCanvas(backgroundMaskImage, CANVAS_NAMES.mask);
  if (edgeBlurAmount === 0) {
    return backgroundMask;
  } else {
    return drawAndBlurImageOnOffScreenCanvas(
        backgroundMask, edgeBlurAmount, CANVAS_NAMES.blurredMask);
  }
}

/**
 * Given a personSegmentation and an image, draws the image with its background
 * blurred onto the canvas.
 *
 * @param canvas The canvas to draw the background-blurred image onto.
 *
 * @param image The image to blur the background of and draw.
 *
 * @param personSegmentation A personSegmentation object, containing a binary
 * array with 1 for the pixels that are part of the person, and 0 otherwise.
 * Must have the same dimensions as the image.
 *
 * @param backgroundBlurAmount How many pixels in the background blend into each
 * other.  Defaults to 3. Should be an integer between 1 and 20.
 *
 * @param edgeBlurAmount How many pixels to blur on the edge between the person
 * and the background by.  Defaults to 3. Should be an integer between 0 and 20.
 *
 * @param flipHorizontal If the output should be flipped horizontally.  Defaults
 * to false.
 */
export function drawBokehEffect(
    canvas: HTMLCanvasElement, image: ImageType,
    personSegmentation: PersonSegmentation, backgroundBlurAmount = 3,
    edgeBlurAmount = 3, flipHorizontal = false) {
  assertSameDimensions(image, personSegmentation, 'image', 'segmentation');

  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
      image, backgroundBlurAmount, CANVAS_NAMES.blurred);

  const backgroundMask =
      createBackgroundMask(personSegmentation, edgeBlurAmount);

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  // draw the original image on the final canvas
  ctx.drawImage(image, 0, 0);
  // "destination-in" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // crop what's not the person using the mask from the original image
  drawWithCompositing(ctx, backgroundMask, 'destination-in');
  // "source-over" - "This is the default setting and draws new shapes on top
  // of the existing canvas content." draw the blurred background on top
  // of that.
  drawWithCompositing(ctx, blurredImage, 'destination-over');
  ctx.restore();
}
