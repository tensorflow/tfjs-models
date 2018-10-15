import * as posenet from '@tensorflow-models/posenet';
import * as tf from '@tensorflow/tfjs';

let offScreenCanvas;

function createOffScreenCanvas() {
  offScreenCanvas = document.createElement('canvas');
  offScreenCanvas.style.display = 'none';
  document.body.append(offScreenCanvas);
}

const blurAmount = 4;

function drawBlurredImageToOffscreenCanvas(image, flipHorizontal) {
  const {height, width} = image;
  const ctx = offScreenCanvas.getContext('2d');
  offScreenCanvas.width = width;
  offScreenCanvas.height = height;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  if (flipHorizontal) {
    ctx.scale(-1, 1);
    ctx.translate(-width, 0);
  }
  ctx.filter = `blur(${blurAmount}px)`;
  ctx.drawImage(video, 0, 0, width, height);
  ctx.restore();
}

export async function applyBokehEffect(
    canvas, input, mask, flipHorizontal = true) {
  if (!offScreenCanvas) {
    createOffScreenCanvas();
  }

  drawBlurredImageToOffscreenCanvas(input, flipHorizontal);

  const bokehedImage = tf.tidy(() => {
    const blurredImage = tf.fromPixels(offScreenCanvas);

    const image =
        flipHorizontal ? tf.fromPixels(input).reverse(1) : tf.fromPixels(input);

    const [height, width] = image.shape;

    const segmentationMask =
        tf.tensor2d(mask, [height, width], 'int32').expandDims(2);

    const invertedMask = tf.scalar(1, 'int32').sub(segmentationMask);

    const blurredImageWithoutPerson = blurredImage.mul(invertedMask);

    const imageWithoutBackground = image.mul(segmentationMask);

    return blurredImageWithoutPerson.add(imageWithoutBackground);
  });

  await tf.toPixels(bokehedImage, canvas);

  bokehedImage.dispose();
}
