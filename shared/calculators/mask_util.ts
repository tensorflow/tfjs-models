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
import * as tf from '@tensorflow/tfjs-core';

function toNumber(value: number|SVGAnimatedLength) {
  return value instanceof SVGAnimatedLength ? value.baseVal.value : value;
}

/**
 * Converts input image to an HTMLCanvasElement. Note that converting
 * back from the output of this function to imageData or a Tensor will be lossy
 * due to premultiplied alpha color values. For more details please reference:
 * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/putImageData#data_loss_due_to_browser_optimization
 * @param image Input image.
 *
 * @returns Converted HTMLCanvasElement.
 */
export async function toHTMLCanvasElementLossy(
    image: ImageData|tf.Tensor2D|tf.Tensor3D|SVGImageElement|
    OffscreenCanvas): Promise<HTMLCanvasElement> {
  const canvas = document.createElement('canvas');

  if (image instanceof tf.Tensor) {
    await tf.browser.toPixels(image, canvas);
  } else {
    canvas.width = toNumber(image.width);
    canvas.height = toNumber(image.height);

    const ctx = canvas.getContext('2d');
    if (image instanceof ImageData) {
      ctx.putImageData(image, 0, 0);
    } else {
      ctx.drawImage(image, 0, 0);
    }
  }

  return canvas;
}

/**
 * Converts input image to ImageData. Note that converting
 * from a CanvasImageSource will be lossy due to premultiplied alpha color
 * values. For more details please reference:
 * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/putImageData#data_loss_due_to_browser_optimization
 * @param image Input image.
 *
 * @returns Converted ImageData.
 */
export async function toImageDataLossy(image: CanvasImageSource|
                                       tf.Tensor3D): Promise<ImageData> {
  if (image instanceof tf.Tensor) {
    const [height, width] = image.shape.slice(0, 2);
    return new ImageData(await tf.browser.toPixels(image), width, height);
  } else {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    canvas.width = toNumber(image.width);
    canvas.height = toNumber(image.height);

    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
}

/**
 * Converts input image to Tensor. Note that converting
 * from a CanvasImageSource will be lossy due to premultiplied alpha color
 * values. For more details please reference:
 * https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/putImageData#data_loss_due_to_browser_optimization
 * @param image Input image.
 *
 * @returns Converted Tensor.
 */
export async function toTensorLossy(image: CanvasImageSource|
                                    ImageData): Promise<tf.Tensor3D> {
  const pixelsInput =
      (image instanceof SVGImageElement || image instanceof OffscreenCanvas) ?
      await toHTMLCanvasElementLossy(image) :
      image;
  return tf.browser.fromPixels(pixelsInput, 4);
}

export function assertMaskValue(maskValue: number) {
  if (maskValue < 0 || maskValue >= 256) {
    throw new Error(
        `Mask value must be in range [0, 255] but got ${maskValue}`);
  }
  if (!Number.isInteger(maskValue)) {
    throw new Error(`Mask value must be an integer but got ${maskValue}`);
  }
}
