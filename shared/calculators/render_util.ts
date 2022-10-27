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

import * as tf from '@tensorflow/tfjs-core';
import {Color, PixelInput, Segmentation} from './interfaces/common_interfaces';

/**
 * This render_util implementation is based on the body-pix output_rending_util
 * code found here:
 * https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/output_rendering_util.ts
 * It is adapted to account for the generic segmentation interface.
 */

type ImageType = CanvasImageSource|OffscreenCanvas|PixelInput;
type HasDimensions = {
  width: number,
  height: number
};

type Canvas = HTMLCanvasElement|OffscreenCanvas;

const CANVAS_NAMES = {
  blurred: 'blurred',
  blurredMask: 'blurred-mask',
  mask: 'mask',
  lowresPartMask: 'lowres-part-mask',
  drawImage: 'draw-image',
};

const offScreenCanvases: {[name: string]: Canvas} = {};

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

function getSizeFromImageLikeElement(input: HTMLImageElement|HTMLCanvasElement|
                                     OffscreenCanvas): [number, number] {
  if ('offsetHeight' in input && input.offsetHeight !== 0 &&
      'offsetWidth' in input && input.offsetWidth !== 0) {
    return [input.offsetHeight, input.offsetWidth];
  } else if (input.height != null && input.width != null) {
    return [input.height, input.width];
  } else {
    throw new Error(
        `HTMLImageElement must have height and width attributes set.`);
  }
}

function getSizeFromVideoElement(input: HTMLVideoElement): [number, number] {
  if (input.hasAttribute('height') && input.hasAttribute('width')) {
    // Prioritizes user specified height and width.
    // We can't test the .height and .width properties directly,
    // because they evaluate to 0 if unset.
    return [input.height, input.width];
  } else {
    return [input.videoHeight, input.videoWidth];
  }
}

function getInputSize(input: ImageType): [number, number] {
  if ((typeof (HTMLCanvasElement) !== 'undefined' &&
       input instanceof HTMLCanvasElement) ||
      (typeof (OffscreenCanvas) !== 'undefined' &&
       input instanceof OffscreenCanvas) ||
      (typeof (HTMLImageElement) !== 'undefined' &&
       input instanceof HTMLImageElement)) {
    return getSizeFromImageLikeElement(input);
  } else if (typeof (ImageData) !== 'undefined' && input instanceof ImageData) {
    return [input.height, input.width];
  } else if (
      typeof (HTMLVideoElement) !== 'undefined' &&
      input instanceof HTMLVideoElement) {
    return getSizeFromVideoElement(input);
  } else if (input instanceof tf.Tensor) {
    return [input.shape[0], input.shape[1]];
  } else {
    throw new Error(`error: Unknown input type: ${input}.`);
  }
}

function createOffScreenCanvas(): Canvas {
  if (typeof document !== 'undefined') {
    return document.createElement('canvas');
  } else if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(0, 0);
  } else {
    throw new Error('Cannot create a canvas in this context');
  }
}

function ensureOffscreenCanvasCreated(id: string): Canvas {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}

/**
 * Draw image data on a canvas.
 */
function renderImageDataToCanvas(image: ImageData, canvas: Canvas) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(image, 0, 0);
}

function renderImageDataToOffScreenCanvas(
    image: ImageData, canvasName: string): Canvas {
  const canvas = ensureOffscreenCanvasCreated(canvasName);
  renderImageDataToCanvas(image, canvas);

  return canvas;
}

/**
 * Draw image on a 2D rendering context.
 */
async function drawImage(
    ctx: CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D,
    image: ImageType, dx: number, dy: number, dw?: number, dh?: number) {
  if (image instanceof tf.Tensor) {
    const pixels = await tf.browser.toPixels(image);
    const [height, width] = getInputSize(image);
    image = new ImageData(pixels, width, height);
  }
  if (image instanceof ImageData) {
    image = renderImageDataToOffScreenCanvas(image, CANVAS_NAMES.drawImage);
  }
  if (dw == null || dh == null) {
    ctx.drawImage(image, dx, dy);
  } else {
    ctx.drawImage(image, dx, dy, dw, dh);
  }
}

/**
 * Draw image on a canvas.
 */
async function renderImageToCanvas(image: ImageType, canvas: Canvas) {
  const [height, width] = getInputSize(image);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  await drawImage(ctx, image, 0, 0, width, height);
}

function flipCanvasHorizontal(canvas: Canvas) {
  const ctx = canvas.getContext('2d');
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
}

async function drawWithCompositing(
    ctx: CanvasRenderingContext2D|OffscreenCanvasRenderingContext2D,
    image: ImageType, compositeOperation: string) {
  // TODO: Assert type 'compositeOperation as GlobalCompositeOperation' after
  // typescript update to 4.6.0 or later
  // tslint:disable-next-line: no-any
  ctx.globalCompositeOperation = compositeOperation as any;
  await drawImage(ctx, image, 0, 0);
}

// method copied from bGlur in https://codepen.io/zhaojun/pen/zZmRQe
async function cpuBlur(canvas: Canvas, image: ImageType, blur: number) {
  const ctx = canvas.getContext('2d');

  let sum = 0;
  const delta = 5;
  const alphaLeft = 1 / (2 * Math.PI * delta * delta);
  const step = blur < 3 ? 1 : 2;
  for (let y = -blur; y <= blur; y += step) {
    for (let x = -blur; x <= blur; x += step) {
      const weight =
          alphaLeft * Math.exp(-(x * x + y * y) / (2 * delta * delta));
      sum += weight;
    }
  }
  for (let y = -blur; y <= blur; y += step) {
    for (let x = -blur; x <= blur; x += step) {
      ctx.globalAlpha = alphaLeft *
          Math.exp(-(x * x + y * y) / (2 * delta * delta)) / sum * blur;
      await drawImage(ctx, image, x, y);
    }
  }
  ctx.globalAlpha = 1;
}

async function drawAndBlurImageOnCanvas(
    image: ImageType, blurAmount: number, canvas: Canvas) {
  const [height, width] = getInputSize(image);
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  if (isSafari()) {
    await cpuBlur(canvas, image, blurAmount);
  } else {
    // tslint:disable:no-any
    (ctx as any).filter = `blur(${blurAmount}px)`;
    await drawImage(ctx, image, 0, 0, width, height);
  }
  ctx.restore();
}

async function drawAndBlurImageOnOffScreenCanvas(
    image: ImageType, blurAmount: number,
    offscreenCanvasName: string): Promise<Canvas> {
  const canvas = ensureOffscreenCanvasCreated(offscreenCanvasName);
  if (blurAmount === 0) {
    await renderImageToCanvas(image, canvas);
  } else {
    await drawAndBlurImageOnCanvas(image, blurAmount, canvas);
  }
  return canvas;
}

function drawStroke(
    bytes: Uint8ClampedArray, row: number, column: number, width: number,
    radius: number, color: Color = {
      r: 0,
      g: 255,
      b: 255,
      a: 255
    }) {
  for (let i = -radius; i <= radius; i++) {
    for (let j = -radius; j <= radius; j++) {
      if (i !== 0 && j !== 0) {
        const n = (row + i) * width + (column + j);
        bytes[4 * n + 0] = color.r;
        bytes[4 * n + 1] = color.g;
        bytes[4 * n + 2] = color.b;
        bytes[4 * n + 3] = color.a;
      }
    }
  }
}

function isSegmentationBoundary(
    data: Uint8ClampedArray,
    row: number,
    column: number,
    width: number,
    isForegroundId: boolean[],
    alphaCutoff: number,
    radius = 1,
    ): boolean {
  let numberBackgroundPixels = 0;
  for (let i = -radius; i <= radius; i++) {
    for (let j = -radius; j <= radius; j++) {
      if (i !== 0 && j !== 0) {
        const n = (row + i) * width + (column + j);
        if (!isForegroundId[data[4 * n]] || data[4 * n + 3] < alphaCutoff) {
          numberBackgroundPixels += 1;
        }
      }
    }
  }
  return numberBackgroundPixels > 0;
}

/**
 * Given a segmentation or array of segmentations, generates an
 * image with foreground and background color at each pixel determined by the
 * corresponding binary segmentation value at the pixel from the output.  In
 * other words, pixels where there is a person will be colored with foreground
 * color and where there is not a person will be colored with background color.
 *
 * @param segmentation Single segmentation or array of segmentations.
 *
 * @param foreground Default to {r:0, g:0, b:0, a: 0}. The foreground color
 * (r,g,b,a) for visualizing pixels that belong to people.
 *
 * @param background Default to {r:0, g:0, b:0, a: 255}. The background color
 * (r,g,b,a) for visualizing pixels that don't belong to people.
 *
 * @param drawContour Default to false. Whether to draw the contour around each
 * person's segmentation mask or body part mask.
 *
 * @param foregroundThreshold Default to 0.5. The minimum probability
 * to color a pixel as foreground rather than background. The alpha channel
 * integer values will be taken as the probabilities (for more information refer
 * to `Segmentation` type's documentation).
 *
 * @param foregroundMaskValues Default to all mask values. The red channel
 *     integer values that represent foreground (for more information refer to
 * `Segmentation` type's documentation).
 *
 * @returns An ImageData with the same width and height of
 * the input segmentations, with opacity and
 * transparency at each pixel determined by the corresponding binary
 * segmentation value at the pixel from the output.
 */
export async function toBinaryMask(
    segmentation: Segmentation|Segmentation[], foreground: Color = {
      r: 0,
      g: 0,
      b: 0,
      a: 0
    },
    background: Color = {
      r: 0,
      g: 0,
      b: 0,
      a: 255
    },
    drawContour = false, foregroundThreshold = 0.5,
    foregroundMaskValues = Array.from(Array(256).keys())) {
  const segmentations =
      !Array.isArray(segmentation) ? [segmentation] : segmentation;

  if (segmentations.length === 0) {
    return null;
  }

  const masks = await Promise.all(
      segmentations.map(segmentation => segmentation.mask.toImageData()));

  const {width, height} = masks[0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const alphaCutoff = Math.round(255 * foregroundThreshold);
  const isForegroundId: boolean[] = new Array(256).fill(false);
  foregroundMaskValues.forEach(id => isForegroundId[id] = true);

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      const n = i * width + j;
      bytes[4 * n + 0] = background.r;
      bytes[4 * n + 1] = background.g;
      bytes[4 * n + 2] = background.b;
      bytes[4 * n + 3] = background.a;
      for (const mask of masks) {
        if (isForegroundId[mask.data[4 * n]] &&
            mask.data[4 * n + 3] >= alphaCutoff) {
          bytes[4 * n] = foreground.r;
          bytes[4 * n + 1] = foreground.g;
          bytes[4 * n + 2] = foreground.b;
          bytes[4 * n + 3] = foreground.a;
          if (drawContour && i - 1 >= 0 && i + 1 < height && j - 1 >= 0 &&
              j + 1 < width &&
              isSegmentationBoundary(
                  mask.data, i, j, width, isForegroundId, alphaCutoff)) {
            drawStroke(bytes, i, j, width, 1);
          }
        }
      }
    }
  }

  return new ImageData(bytes, width, height);
}

/**
 * Given a segmentation or array of segmentations, and a function mapping
 * the red pixel values (representing body part labels) to colours,
 * generates an image with the corresponding color for each part at each pixel,
 * and background color used where there is no part.
 *
 * @param segmentation Single segmentation or array of segmentations.
 *
 * @param maskValueToColor A function mapping red channel mask values to
 * colors to use in output image.
 *
 * @param background Default to {r:0, g:0, b:0, a: 255}. The background color
 * (r,g,b,a) for visualizing pixels that don't belong to people.
 *
 * @param foregroundThreshold Default to 0.5. The minimum probability
 * to color a pixel as foreground rather than background. The alpha channel
 * integer values will be taken as the probabilities (for more information refer
 * to `Segmentation` type's documentation).
 *
 * @returns An ImageData with the same width and height of input segmentations,
 * with the corresponding color for each part at each pixel, and background
 * pixels where there is no part.
 */
export async function toColoredMask(
    segmentation: Segmentation|Segmentation[],
    maskValueToColor: (maskValue: number) => Color, background: Color = {
      r: 0,
      g: 0,
      b: 0,
      a: 255
    },
    foregroundThreshold = 0.5) {
  const segmentations =
      !Array.isArray(segmentation) ? [segmentation] : segmentation;

  if (segmentations.length === 0) {
    return null;
  }

  const masks = await Promise.all(
      segmentations.map(segmentation => segmentation.mask.toImageData()));

  const {width, height} = masks[0];
  const bytes = new Uint8ClampedArray(width * height * 4);
  const alphaCutoff = Math.round(255 * foregroundThreshold);

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    bytes[j + 0] = background.r;
    bytes[j + 1] = background.g;
    bytes[j + 2] = background.b;
    bytes[j + 3] = background.a;
    for (const mask of masks) {
      if (mask.data[j + 3] >= alphaCutoff) {
        const maskValue = mask.data[j];
        const color = maskValueToColor(maskValue);

        bytes[j + 0] = color.r;
        bytes[j + 1] = color.g;
        bytes[j + 2] = color.b;
        bytes[j + 3] = color.a;
      }
    }
  }

  return new ImageData(bytes, width, height);
}

/**
 * Given an image and a maskImage of type ImageData, draws the image with the
 * mask on top of it onto a canvas.
 *
 * @param canvas The canvas to be drawn onto.
 *
 * @param image The original image to apply the mask to.
 *
 * @param maskImage An ImageData containing the mask. Ideally this should be
 * generated by toBinaryMask or toColoredMask.
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
export async function drawMask(
    canvas: Canvas, image: ImageType, maskImage: ImageData|null,
    maskOpacity = 0.7, maskBlurAmount = 0, flipHorizontal = false) {
  const [height, width] = getInputSize(image);
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }

  await drawImage(ctx, image, 0, 0);

  ctx.globalAlpha = maskOpacity;
  if (maskImage) {
    assertSameDimensions({width, height}, maskImage, 'image', 'mask');

    const mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);

    const blurredMask = await drawAndBlurImageOnOffScreenCanvas(
        mask, maskBlurAmount, CANVAS_NAMES.blurredMask);
    ctx.drawImage(blurredMask, 0, 0, width, height);
  }
  ctx.restore();
}

/**
 * Given an image and a maskImage of type ImageData, draws the image with the
 * pixelated mask on top of it onto a canvas.
 *
 * @param canvas The canvas to be drawn onto.
 *
 * @param image The original image to apply the mask to.
 *
 * @param maskImage An ImageData containing the mask.  Ideally this should be
 * generated by toColoredmask.
 *
 * @param maskOpacity The opacity of the mask when drawing it on top of the
 * image. Defaults to 0.7. Should be a float between 0 and 1.
 *
 * @param maskBlurAmount How many pixels to blur the mask by. Defaults to 0.
 * Should be an integer between 0 and 20.
 *
 * @param flipHorizontal If the result should be flipped horizontally.  Defaults
 * to false.
 *
 * @param pixelCellWidth The width of each pixel cell. Default to 10 px.
 */
export async function drawPixelatedMask(
    canvas: Canvas, image: ImageType, maskImage: ImageData, maskOpacity = 0.7,
    maskBlurAmount = 0, flipHorizontal = false, pixelCellWidth = 10.0) {
  const [height, width] = getInputSize(image);
  assertSameDimensions({width, height}, maskImage, 'image', 'mask');

  const mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);
  const blurredMask = await drawAndBlurImageOnOffScreenCanvas(
      mask, maskBlurAmount, CANVAS_NAMES.blurredMask);

  canvas.width = blurredMask.width;
  canvas.height = blurredMask.height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }

  const offscreenCanvas =
      ensureOffscreenCanvasCreated(CANVAS_NAMES.lowresPartMask);
  const offscreenCanvasCtx = offscreenCanvas.getContext('2d');
  offscreenCanvas.width = blurredMask.width * (1.0 / pixelCellWidth);
  offscreenCanvas.height = blurredMask.height * (1.0 / pixelCellWidth);
  offscreenCanvasCtx.drawImage(
      blurredMask, 0, 0, blurredMask.width, blurredMask.height, 0, 0,
      offscreenCanvas.width, offscreenCanvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
      offscreenCanvas, 0, 0, offscreenCanvas.width, offscreenCanvas.height, 0,
      0, canvas.width, canvas.height);

  // Draws vertical grid lines that are `pixelCellWidth` apart from each other.
  for (let i = 0; i < offscreenCanvas.width; i++) {
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.moveTo(pixelCellWidth * i, 0);
    ctx.lineTo(pixelCellWidth * i, canvas.height);
    ctx.stroke();
  }

  // Draws horizontal grid lines that are `pixelCellWidth` apart from each
  // other.
  for (let i = 0; i < offscreenCanvas.height; i++) {
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.moveTo(0, pixelCellWidth * i);
    ctx.lineTo(canvas.width, pixelCellWidth * i);
    ctx.stroke();
  }

  ctx.globalAlpha = 1.0 - maskOpacity;
  await drawImage(ctx, image, 0, 0, blurredMask.width, blurredMask.height);
  ctx.restore();
}

async function createPersonMask(
    segmentation: Segmentation|Segmentation[], foregroundThreshold: number,
    edgeBlurAmount: number): Promise<Canvas> {
  const backgroundMaskImage = await toBinaryMask(
      segmentation, {r: 0, g: 0, b: 0, a: 255}, {r: 0, g: 0, b: 0, a: 0}, false,
      foregroundThreshold);

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
 * Given a segmentation or array of segmentations, and an image, draws the image
 * with its background blurred onto the canvas.
 *
 * @param canvas The canvas to draw the background-blurred image onto.
 *
 * @param image The image to blur the background of and draw.
 *
 * @param segmentation Single segmentation or array of segmentations.
 *
 * @param foregroundThreshold Default to 0.5. The minimum probability
 * to color a pixel as foreground rather than background. The alpha channel
 * integer values will be taken as the probabilities (for more information refer
 * to `Segmentation` type's documentation).
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
export async function drawBokehEffect(
    canvas: Canvas, image: ImageType, segmentation: Segmentation|Segmentation[],
    foregroundThreshold = 0.5, backgroundBlurAmount = 3, edgeBlurAmount = 3,
    flipHorizontal = false) {
  const blurredImage = await drawAndBlurImageOnOffScreenCanvas(
      image, backgroundBlurAmount, CANVAS_NAMES.blurred);
  canvas.width = blurredImage.width;
  canvas.height = blurredImage.height;

  const ctx = canvas.getContext('2d');

  if (Array.isArray(segmentation) && segmentation.length === 0) {
    ctx.drawImage(blurredImage, 0, 0);
    return;
  }

  const personMask =
      await createPersonMask(segmentation, foregroundThreshold, edgeBlurAmount);

  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  // draw the original image on the final canvas
  const [height, width] = getInputSize(image);
  await drawImage(ctx, image, 0, 0, width, height);

  // "destination-in" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // crop what's not the person using the mask from the original image
  await drawWithCompositing(ctx, personMask, 'destination-in');
  // "destination-over" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // draw the blurred background on top of the original image where it doesn't
  // overlap.
  await drawWithCompositing(ctx, blurredImage, 'destination-over');
  ctx.restore();
}

async function createBodyPartMask(
    segmentation: Segmentation|Segmentation[], maskValuesToBlur: number[],
    foregroundThreshold: number, edgeBlurAmount: number): Promise<Canvas> {
  const backgroundMaskImage = await toBinaryMask(
      segmentation, {r: 0, g: 0, b: 0, a: 0}, {r: 0, g: 0, b: 0, a: 255}, true,
      foregroundThreshold, maskValuesToBlur);

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
 * @param segmentation Single segmentation or array of segmentations.
 *
 * @param maskValuesToBlur An array of red channel mask values to blur
 *     (representing different body parts, refer to `Segmentation` interface
 * docs for more details).
 *
 * @param foregroundThreshold Default to 0.5. The minimum probability
 * to color a pixel as foreground rather than background. The alpha channel
 * integer values will be taken as the probabilities (for more information refer
 * to `Segmentation` type's documentation).
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
export async function blurBodyPart(
    canvas: Canvas, image: ImageType, segmentation: Segmentation|Segmentation[],
    maskValuesToBlur: number[], foregroundThreshold = 0.5,
    backgroundBlurAmount = 3, edgeBlurAmount = 3, flipHorizontal = false) {
  const blurredImage = await drawAndBlurImageOnOffScreenCanvas(
      image, backgroundBlurAmount, CANVAS_NAMES.blurred);
  canvas.width = blurredImage.width;
  canvas.height = blurredImage.height;

  const ctx = canvas.getContext('2d');

  if (Array.isArray(segmentation) && segmentation.length === 0) {
    ctx.drawImage(blurredImage, 0, 0);
    return;
  }
  const bodyPartMask = await createBodyPartMask(
      segmentation, maskValuesToBlur, foregroundThreshold, edgeBlurAmount);

  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }
  // draw the original image on the final canvas
  const [height, width] = getInputSize(image);
  await drawImage(ctx, image, 0, 0, width, height);

  // "destination-in" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // crop what's not the person using the mask from the original image
  await drawWithCompositing(ctx, bodyPartMask, 'destination-in');
  // "destination-over" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // draw the blurred background on top of the original image where it doesn't
  // overlap.
  await drawWithCompositing(ctx, blurredImage, 'destination-over');
  ctx.restore();
}
