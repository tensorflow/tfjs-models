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
import {Color, PartSegmentation, PersonSegmentation} from './types';

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
 * Given the output from estimating single-person segmentation, generates an
 * image with foreground and background color at each pixel determined by the
 * corresponding binary segmentation value at the pixel from the output.  In
 * other words, pixels where there is a person will be colored with foreground
 * color and where there is not a person will be colored with background color.
 *
 * @param segmentation The output from estimagePersonSegmentation; an object
 * containing a width, height, and a binary array with 1 for the pixels that are
 * part of the person, and 0 otherwise.
 *
 * @param foreground The foreground color (r,g,b,a) for visualizing pixels that
 * belong to people.
 *
 * @param background The background color (r,g,b,a) for visualizing pixels that
 * don't belong to people.
 *
 * @param drawContour Whether to draw the contour around each person's
 * segmentation mask.
 *
 * @returns An ImageData with the same width and height of the
 * segmentation, with opacity and transparency at each pixel determined by
 * the corresponding binary segmentation value at the pixel from the output.
 */
export function toMaskImageData(
    segmentation: PersonSegmentation, foreground: Color = {
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
    drawContour = false): ImageData {
  return toMultiPersonMaskImageData(
      [segmentation], foreground, background, drawContour);
}

/**
 * Given the output from estimating multi-person segmentation, generates an
 * image with foreground and background color at each pixel determined by the
 * corresponding binary segmentation value at the pixel from the output.  In
 * other words, pixels where there is a person will be colored with foreground
 * color and where there is not a person will be colored with background color.
 *
 * @param multiPersonSegmentation The output from estimateMultiPersonSegmentation;
 * An array of PersonSegmentation object, each containing a width, height, and a
 * binary array with 1 for the pixels that are part of the person, and 0
 * otherwise.
 *
 * @param foreground The foreground color (r,g,b,a) for visualizing pixels that
 * belong to people.
 *
 * @param background The background color (r,g,b,a) for visualizing pixels that
 * don't belong to people.
 *
 * @param drawContour Whether to draw the contour around each person's
 * segmentation mask.
 *
 * @returns An ImageData with the same width and height of
 * all the PersonSegmentation in multiPersonSegmentation, with opacity and
 * transparency at each pixel determined by the corresponding binary
 * segmentation value at the pixel from the output.
 */
export function toMultiPersonMaskImageData(
    multiPersonSegmentation: PersonSegmentation[], foreground: Color = {
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
    drawContour = false): ImageData|null {
  if (multiPersonSegmentation.length === 0) {
    return null;
  }

  const {width, height} = multiPersonSegmentation[0];
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height; i += 1) {
    for (let j = 0; j < width; j += 1) {
      const n = i * width + j;
      bytes[4 * n + 0] = background.r;
      bytes[4 * n + 1] = background.g;
      bytes[4 * n + 2] = background.b;
      bytes[4 * n + 3] = background.a;
      for (let k = 0; k < multiPersonSegmentation.length; k++) {
        if (multiPersonSegmentation[k].data[n] === 1) {
          bytes[4 * n] = foreground.r;
          bytes[4 * n + 1] = foreground.g;
          bytes[4 * n + 2] = foreground.b;
          bytes[4 * n + 3] = foreground.a;

          if (drawContour) {
            // checks boundary
            if (i - 1 >= 0 && i + 1 < height && j - 1 >= 0 && j + 1 < width) {
              const n11 = (i - 1) * width + j - 1;
              const n12 = (i - 1) * width + j;
              const n13 = (i - 1) * width + j + 1;
              const n21 = i * width + j - 1;
              const n23 = i * width + j + 1;
              const n31 = (i + 1) * width + j - 1;
              const n32 = (i + 1) * width + j;
              const n33 = (i + 1) * width + j + 1;
              if (multiPersonSegmentation[k].data[n11] !== 1 ||
                  multiPersonSegmentation[k].data[n12] !== 1 ||
                  multiPersonSegmentation[k].data[n13] !== 1 ||
                  multiPersonSegmentation[k].data[n21] !== 1 ||
                  multiPersonSegmentation[k].data[n23] !== 1 ||
                  multiPersonSegmentation[k].data[n31] !== 1 ||
                  multiPersonSegmentation[k].data[n32] !== 1 ||
                  multiPersonSegmentation[k].data[n33] !== 1) {
                for (let nn
                         of [n, n11, n12, n13, n21, n23, n31, n31, n32, n33]) {
                  bytes[4 * nn + 0] = 0;
                  bytes[4 * nn + 1] = 255;
                  bytes[4 * nn + 2] = 255;
                }
              }
            }
          }
        }
      }
    }
  }

  return new ImageData(bytes, width, height);
}

/**
 * Given the output from estimating single-person part segmentation, and an
 * array of colors indexed by part id, generates an image with the corresponding
 * color for each part at each pixel, and white pixels where there is no part.
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
  return toMultiPersonColoredPartImageData([partSegmentation], partColors);
}

/**
 * Given the output from estimating multi-person part segmentation, and an array
 * of colors indexed by part id, generates an image with the corresponding color
 * for each part at each pixel, and white pixels where there is no part.
 *
 * @param allPartSegmentation The output from
 * estimateMultiPersonPartSegmentation; an array of PartSegmentation object
 * containing a width, height, and an array with a part id from 0-24 for the
 * pixels that are part of a corresponding body part, and -1 otherwise.
 *
 * @param partColors A multi-dimensional array of rgb colors indexed by
 * part id.  Must have 24 colors, one for every part.
 *
 * @returns An ImageData with the same width and height of all the element in
 * allPartSegmentation, with the corresponding color for each part at each
 * pixel, and black pixels where there is no part.
 */
export function toMultiPersonColoredPartImageData(
    multiPersonSegmentation: PartSegmentation[],
    partColors: Array<[number, number, number]>): ImageData {
  const {width, height} = multiPersonSegmentation[0];
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    // const partId = Math.round(data[i]);
    const j = i * 4;
    bytes[j + 0] = 255;
    bytes[j + 1] = 255;
    bytes[j + 2] = 255;
    bytes[j + 3] = 255;
    for (let k = 0; k < multiPersonSegmentation.length; k++) {
      const partId = multiPersonSegmentation[k].data[i];
      if (partId !== -1) {
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
  }
  return new ImageData(bytes, width, height);
}

const CANVAS_NAMES = {
  blurred: 'blurred',
  blurredMask: 'blurred-mask',
  mask: 'mask',
  lowresPartMask: 'lowres-part-mask',
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
    canvas: HTMLCanvasElement, image: ImageType, maskImage: ImageData|null,
    maskOpacity = 0.7, maskBlurAmount = 0, flipHorizontal = false) {
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    flipCanvasHorizontal(canvas);
  }

  ctx.drawImage(image, 0, 0);
  ctx.globalAlpha = maskOpacity;
  if (maskImage) {
    assertSameDimensions(image, maskImage, 'image', 'mask');

    const mask = renderImageDataToOffScreenCanvas(maskImage, CANVAS_NAMES.mask);

    const blurredMask = drawAndBlurImageOnOffScreenCanvas(
        mask, maskBlurAmount, CANVAS_NAMES.blurredMask);
    ctx.drawImage(blurredMask, 0, 0);
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
 * generated by toColoredPartImageData.
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
export function drawPixelatedMask(
    canvas: HTMLCanvasElement, image: ImageType, maskImage: ImageData,
    maskOpacity = 0.7, maskBlurAmount = 0, flipHorizontal = false,
    pixelCellWidth = 10.0) {
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
  ctx.drawImage(image, 0, 0);
  ctx.restore();
}

function createPersonMask(
    multiPersonSegmentations: PersonSegmentation[],
    edgeBlurAmount: number): HTMLCanvasElement {
  const backgroundMaskImage =
      toMultiPersonMaskImageData(multiPersonSegmentations);

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
  // assertSameDimensions(image, personSegmentation, 'image', 'segmentation');

  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
      image, backgroundBlurAmount, CANVAS_NAMES.blurred);

  const personMask = createPersonMask([personSegmentation], edgeBlurAmount);

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
  drawWithCompositing(ctx, personMask, 'destination-in');
  // "destination-over" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // draw the blurred background on top of the original image where it doesn't
  // overlap.
  drawWithCompositing(ctx, blurredImage, 'destination-over');
  ctx.restore();
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
export function drawMultiPersonBokehEffect(
    canvas: HTMLCanvasElement, image: ImageType,
    multiPersonSegmentation: PersonSegmentation[], backgroundBlurAmount = 3,
    edgeBlurAmount = 3, flipHorizontal = false) {
  // assertSameDimensions(image, personSegmentation, 'image', 'segmentation');

  const blurredImage = drawAndBlurImageOnOffScreenCanvas(
      image, backgroundBlurAmount, CANVAS_NAMES.blurred);

  const personMask = createPersonMask(multiPersonSegmentation, edgeBlurAmount);

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
  drawWithCompositing(ctx, personMask, 'destination-in');
  // "destination-over" - "The existing canvas content is kept where both the
  // new shape and existing canvas content overlap. Everything else is made
  // transparent."
  // draw the blurred background on top of the original image where it doesn't
  // overlap.
  drawWithCompositing(ctx, blurredImage, 'destination-over');
  ctx.restore();
}
