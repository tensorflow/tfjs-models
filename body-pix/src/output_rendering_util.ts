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

type ImageType = HTMLImageElement|HTMLVideoElement;
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
    compostOperation: string) {
  ctx.globalCompositeOperation = compostOperation;
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

/**
 * Draw an image on a canvas
 */
function drawImageOnCanvas(image: ImageData, canvas: HTMLCanvasElement) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(image, 0, 0);
}

export function toMaskImageData(
    segmentation: PersonSegmentation, invertMask: boolean,
    darknessLevel = 0.7): ImageData {
  const {width, height, data} = segmentation;
  const bytes = new Uint8ClampedArray(width * height * 4);

  const multiplier = Math.round(255 * darknessLevel);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const shouldMask = invertMask ? 1 - data[i] : data[i];
    // alpha will determine how dark the mask should be.
    const alpha = shouldMask * multiplier;

    const j = i * 4;
    bytes[j + 0] = 0;
    bytes[j + 1] = 0;
    bytes[j + 2] = 0;
    bytes[j + 3] = Math.round(alpha);
  }

  return new ImageData(bytes, width, height);
}

export function drawImageWithMaskOnCanvas(
    canvas: HTMLCanvasElement, image: ImageType, mask: ImageData,
    flipHorizontal = true) {
  assertSameDimensions(image, mask, 'image', 'mask');

  const maskCanvas = ensureOffscreenCanvasCreated('mask');
  drawImageOnCanvas(mask, maskCanvas);

  canvas.width = mask.width;
  canvas.height = mask.height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }

  ctx.drawImage(image, 0, 0);
  // 'source-atop' - 'The new shape is only drawn where it overlaps the
  // existing canvas content.'
  drawWithCompositing(ctx, maskCanvas, 'source-over');
  ctx.restore();
}

export function drawBokehEffectOnCanvas(
    canvas: HTMLCanvasElement, image: ImageType,
    segmentation: PersonSegmentation, bokehBlurAmount = 3,
    flipHorizontal = true) {
  assertSameDimensions(image, segmentation, 'image', 'segmentation');

  const blurredCanvas = ensureOffscreenCanvasCreated('blur');

  drawAndBlurImageOnCanvas(image, bokehBlurAmount, blurredCanvas);

  const invertMask = false;
  const darknessLevel = 1.;

  const backgroundMaskImage =
      toMaskImageData(segmentation, invertMask, darknessLevel);

  const backgroundMask = ensureOffscreenCanvasCreated('mask');
  drawImageOnCanvas(backgroundMaskImage, backgroundMask);

  const blurredCtx = blurredCanvas.getContext('2d');
  blurredCtx.save();
  // "destination-out" - "The existing content is kept where it doesn't
  // overlap the new shape." crop person using the mask from the blurred image
  drawWithCompositing(blurredCtx, backgroundMask, 'destination-out');
  blurredCtx.restore();

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
  drawWithCompositing(ctx, blurredCanvas, 'source-over');
  ctx.restore();
}

export function toColoredPartImageData(
    partSegmentation: PartSegmentation,
    partColors: Array<[number, number, number]>, alpha: number): ImageData {
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
      bytes[j + 3] = Math.round(255 * alpha);
    } else {
      const color = partColors[partId];

      if (!color) {
        throw new Error(`No color could be found for part id ${partId}`);
      }
      bytes[j + 0] = color[0];
      bytes[j + 1] = color[1];
      bytes[j + 2] = color[2];
      bytes[j + 3] = Math.round(255 * alpha);
    }
  }

  return new ImageData(bytes, width, height);
}

export function drawPersonWithBackgroundRemoved(
    canvas: HTMLCanvasElement, image: ImageType,
    segmentation: PersonSegmentation, flipHorizontal = true) {
  assertSameDimensions(image, segmentation, 'image', 'segmentation');

  canvas.width = image.width;
  canvas.height = image.height;

  const invertMask = false;
  const darknessLevel = 1.0;
  const personMaskImageData =
      toMaskImageData(segmentation, invertMask, darknessLevel);
  const personMask = ensureOffscreenCanvasCreated('person-mask');
  drawImageOnCanvas(personMaskImageData, personMask);

  const ctx = canvas.getContext('2d');
  ctx.save();
  canvas.width = segmentation.width;
  canvas.height = segmentation.height;
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  // draw the original image on the final canvas
  ctx.drawImage(image, 0, 0);
  // "destination-in" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // crop what's not the person using the mask from the original image
  drawWithCompositing(ctx, personMask, 'destination-in');
  ctx.restore();
}

export function drawPersonWithBackgroundReplaced(
    canvas: HTMLCanvasElement, image: ImageType,
    segmentation: PersonSegmentation, newBackground: ImageType,
    flipHorizontal = true) {
  assertSameDimensions(image, segmentation, 'image', 'segmentation');
  assertSameDimensions(image, newBackground, 'image', 'newBackground');

  canvas.width = image.width;
  canvas.height = image.height;

  drawPersonWithBackgroundRemoved(canvas, image, segmentation, flipHorizontal);

  const ctx = canvas.getContext('2d');
  ctx.save();
  // "source-out" - The new shape is drawn where it doesn't overlap the existing
  // canvas content."
  // draw the background where it doesnt overlap the cropped person
  drawWithCompositing(ctx, newBackground, 'source-out');

  ctx.restore();
}
