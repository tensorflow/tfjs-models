import * as tf from '@tensorflow/tfjs-core';

export class Box {
  public startEndTensor: tf.Tensor;
  public startPoint: tf.Tensor;
  public endPoint: tf.Tensor;

  constructor(startEndTensor: tf.Tensor) {
    // keep tensor for the next frame
    this.startEndTensor = tf.keep(startEndTensor);
    // startEndTensor[:, 0:2]
    this.startPoint = tf.keep(tf.slice(startEndTensor, [0, 0], [-1, 2]));
    // startEndTensor[:, 2:4]
    this.endPoint = tf.keep(tf.slice(startEndTensor, [0, 2], [-1, 2]));
  }

  getSize() {
    return tf.abs(tf.sub(this.endPoint, this.startPoint));
  }

  getCenter() {
    const halfSize = tf.div(tf.sub(this.endPoint, this.startPoint), 2);
    return tf.add(this.startPoint, halfSize);
  }

  cutFromAndResize(image: tf.Tensor4D, cropSize: [number, number]) {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    const yxyx = tf.concat2d(
        [
          xyxy.slice([0, 1], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 0], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 3], [-1, 1]) as tf.Tensor2D,
          xyxy.slice([0, 2], [-1, 1]) as tf.Tensor2D
        ],
        0);
    const roundedCoords = tf.div(yxyx.transpose(), [h, w, h, w]);
    return tf.image.cropAndResize(
        image, roundedCoords as tf.Tensor2D, [0], cropSize);
  }

  scale(factors: tf.Tensor) {
    const starts = tf.mul(this.startPoint, factors);
    const ends = tf.mul(this.endPoint, factors);

    const newCoordinates =
        tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);
    return new Box(newCoordinates);
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const newSize = tf.mul(tf.div(size, 2), ratio);

    const newStarts = tf.sub(centers, newSize);
    const newEnds = tf.add(centers, newSize);

    return new Box(
        tf.concat2d([newStarts as tf.Tensor2D, newEnds as tf.Tensor2D], 1));
  }
}
