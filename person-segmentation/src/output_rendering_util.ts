const offScreenCanvases: {[name: string]: HTMLCanvasElement} = {};

type ImageType = HTMLImageElement|HTMLVideoElement;

function createOffScreenCanvas(): HTMLCanvasElement {
  const offScreenCanvas = document.createElement('canvas');
  offScreenCanvas.style.display = 'none';
  document.body.appendChild(offScreenCanvas);
  return offScreenCanvas;
}

function ensureOffscreenCanvasCreated(id: string) {
  if (!offScreenCanvases[id]) {
    offScreenCanvases[id] = createOffScreenCanvas();
  }
  return offScreenCanvases[id];
}

function drawBlurredImageToOffscreenCanvas(
    image: ImageType, bokehBlurAmount: number) {
  const blurredCanvas = ensureOffscreenCanvasCreated('blur');
  const {height, width} = image;
  const ctx = blurredCanvas.getContext('2d');
  blurredCanvas.width = width;
  blurredCanvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  // tslint:disable-next-line:no-any
  (ctx as any).filter = `blur(${bokehBlurAmount}px)`;
  ctx.drawImage(image, 0, 0, width, height);
  ctx.restore();

  return blurredCanvas;
}

/**
 * Draw an image on a canvas
 */
export function renderImageToCanvas(
    image: ImageData, width: number, height: number,
    canvas: HTMLCanvasElement) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.putImageData(image, 0, 0);
}

function toMaskImageData(
    mask: Uint8Array, height: number, width: number, invertMask: boolean,
    darknessLevel = 0.7): ImageData {
  const bytes = new Uint8ClampedArray(width * height * 4);

  const multiplier = Math.round(255 * darknessLevel);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const shouldMask = invertMask ? 1 - mask[i] : mask[i];
    const r = 0;
    const g = 0;
    const b = 0;
    // alpha will determine how dark the mask should be.
    const a = shouldMask * multiplier;

    const j = i * 4;
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  return new ImageData(bytes, width, height);
}

function compostMaskOnImage(
    canvas: HTMLCanvasElement, image: ImageType, maskCanvas: HTMLCanvasElement,
    flipHorizontal: boolean) {
  const {height, width} = image;
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
  }
  ctx.drawImage(image, 0, 0);
  ctx.globalCompositeOperation = 'source-atop';
  ctx.drawImage(maskCanvas, 0, 0);
  ctx.restore();
}

export function drawBodyMaskOnCanvas(
    image: ImageType, segmentation: Uint8Array, canvas: HTMLCanvasElement,
    flipHorizontal = true) {
  const {height, width} = image;

  const invertMask = true;

  const maskImage = toMaskImageData(segmentation, height, width, invertMask);

  const maskCanvas = ensureOffscreenCanvasCreated('mask');

  renderImageToCanvas(maskImage, width, height, maskCanvas);

  compostMaskOnImage(canvas, image, maskCanvas, flipHorizontal);
}

export function drawBokehEffectOnCanvas(
    canvas: HTMLCanvasElement, image: ImageType, segmentation: Uint8Array,
    bokehBlurAmount = 3, flipHorizontal = true) {
  const {height, width} = image;
  const blurredCanvas =
      drawBlurredImageToOffscreenCanvas(image, bokehBlurAmount);

  const invertMask = false;
  const darknessLevel = 1.;

  const maskImage =
      toMaskImageData(segmentation, height, width, invertMask, darknessLevel);

  const maskCanvas = ensureOffscreenCanvasCreated('mask');
  renderImageToCanvas(maskImage, width, height, maskCanvas);

  const blurredCtx = blurredCanvas.getContext('2d');
  blurredCtx.save();
  // crop person using the mask from the blurred image
  blurredCtx.globalCompositeOperation = 'destination-out';
  blurredCtx.drawImage(maskCanvas, 0, 0);
  blurredCtx.restore();

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
  }
  // draw the original image on the final canvas
  ctx.drawImage(image, 0, 0);
  // crop what's not the person using the mask from the blurred image
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(maskCanvas, 0, 0);
  //  draw the blurred image without the person on top of the image
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(blurredCanvas, 0, 0);
  ctx.restore();
}

function toColoredPartImage(
    partSegmentation: Int32Array, partColors: Array<[number, number, number]>,
    width: number, height: number): ImageData {
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    // invert mask.  Invert the segmentatino mask.
    const partId = Math.round(partSegmentation[i]);
    const j = i * 4;

    if (partId === -1) {
      bytes[j + 0] = 0;
      bytes[j + 1] = 0;
      bytes[j + 2] = 0;
      bytes[j + 3] = Math.round(255 * 0.7);
    } else {
      const color = partColors[partId];
      if (!color) {
        console.log('no color found', partId);
      }
      bytes[j + 0] = color[0];
      bytes[j + 1] = color[1];
      bytes[j + 2] = color[2];
      bytes[j + 3] = Math.round(255 * .7);
    }
  }

  return new ImageData(bytes, width, height);
}

export function drawBodySegmentsOnCanvas(
    canvas: HTMLCanvasElement, input: ImageType, partSegmentation: Int32Array,
    partColors: Array<[number, number, number]>, partMapDarkening = 0.3,
    flipHorizontal = true) {
  const {height, width} = input;
  canvas.width = width;
  canvas.height = height;
  const coloredPartImage: ImageData =
      toColoredPartImage(partSegmentation, partColors, width, height);

  const partImageCanvas = ensureOffscreenCanvasCreated('partImage');
  partImageCanvas.width = width;
  partImageCanvas.height = height;

  partImageCanvas.getContext('2d').putImageData(coloredPartImage, 0, 0);

  const ctx = canvas.getContext('2d');
  ctx.save();
  if (flipHorizontal) {
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
  }
  ctx.drawImage(input, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(partImageCanvas, 0, 0);
  ctx.restore();
}
