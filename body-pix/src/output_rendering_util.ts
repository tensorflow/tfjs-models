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

function isSafari() {
  return (/^((?!chrome|android).)*safari/i.test(navigator.userAgent));
}

function flipCanvasHorizontal(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
}

function compost(
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

function drawBlurredImageToCanvas(
    image: ImageType, bokehBlurAmount: number, canvas: HTMLCanvasElement) {
  const {height, width} = image;
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  if (isSafari()) {
    cpuBlur(canvas, image, bokehBlurAmount);
  } else {
    // tslint:disable:no-any
    (ctx as any).filter = `blur(${bokehBlurAmount}px)`;
    ctx.drawImage(image, 0, 0, width, height);
  }
  ctx.restore();
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(
    image: ImageData, canvas: HTMLCanvasElement) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(image, 0, 0);
}

export function toMaskImageData(
    mask: Uint8Array, height: number, width: number, invertMask: boolean,
    darknessLevel = 0.7): ImageData {
  const bytes = new Uint8ClampedArray(width * height * 4);

  const multiplier = Math.round(255 * darknessLevel);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const shouldMask = invertMask ? 1 - mask[i] : mask[i];
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

export function drawBodyMaskOnCanvas(
    image: ImageType, segmentation: PersonSegmentation,
    canvas: HTMLCanvasElement, flipHorizontal = true) {
  const {height, width} = segmentation;

  const invertMask = true;

  const maskImage =
      toMaskImageData(segmentation.data, height, width, invertMask);

  const maskCanvas = ensureOffscreenCanvasCreated('mask');

  renderImageToCanvas(maskImage, maskCanvas);

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  ctx.drawImage(image, 0, 0);
  // 'source-atop' - 'The new shape is only drawn where it overlaps the
  // existing canvas content.'
  compost(ctx, maskCanvas, 'source-atop');
  ctx.restore();
}

export function drawBokehEffectOnCanvas(
    canvas: HTMLCanvasElement, image: ImageType,
    segmentation: PersonSegmentation, bokehBlurAmount = 3,
    flipHorizontal = true) {
  const {height, width} = segmentation;
  const blurredCanvas = ensureOffscreenCanvasCreated('blur');

  drawBlurredImageToCanvas(image, bokehBlurAmount, blurredCanvas);

  const invertMask = false;
  const darknessLevel = 1.;

  const invertedMaskImage = toMaskImageData(
      segmentation.data, height, width, invertMask, darknessLevel);

  const maskWhatsNotThePersonCanvas = ensureOffscreenCanvasCreated('mask');
  renderImageToCanvas(invertedMaskImage, maskWhatsNotThePersonCanvas);

  const blurredCtx = blurredCanvas.getContext('2d');
  blurredCtx.save();
  // "destination-out" - "The existing content is kept where it doesn't
  // overlap the new shape." crop person using the mask from the blurred image
  compost(blurredCtx, maskWhatsNotThePersonCanvas, 'destination-out');
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
  // crop what's not the person using the mask from the blurred image
  compost(ctx, maskWhatsNotThePersonCanvas, 'destination-in');
  // "source-over" - "This is the default setting and draws new shapes on top
  // of the existing canvas content." draw the blurred image without the
  // person on top of the image with the person
  compost(ctx, blurredCanvas, 'source-over');
  ctx.restore();
}

function toColoredPartImage(
    partSegmentation: Int32Array, partColors: Array<[number, number, number]>,
    width: number, height: number, alpha: number): ImageData {
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const partId = Math.round(partSegmentation[i]);
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

export function drawBodySegmentsOnCanvas(
    canvas: HTMLCanvasElement, input: ImageType,
    partSegmentation: PartSegmentation,
    partColors: Array<[number, number, number]>, coloredPartImageAlpha = 0.7,
    flipHorizontal = true) {
  const {height, width} = partSegmentation;
  canvas.width = width;
  canvas.height = height;
  // tslint:disable-next-line:no-debugger
  const coloredPartImage: ImageData = toColoredPartImage(
      partSegmentation.data, partColors, width, height, coloredPartImageAlpha);

  const partImageCanvas = ensureOffscreenCanvasCreated('partImage');

  renderImageToCanvas(coloredPartImage, partImageCanvas);

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  ctx.drawImage(input, 0, 0);
  // "source-over: "draws new shapes on top of the existing canvas content."
  compost(ctx, partImageCanvas, 'source-over');
  ctx.restore();
}
