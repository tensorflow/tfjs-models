import * as tf from '@tensorflow/tfjs-core';

export class Box {
  constructor(startEndTensor) {
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

  cutFromAndResize(image, crop_size) {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    const yxyx = tf.concat2d([
      xyxy.slice([0, 1], [-1, 1]), xyxy.slice([0, 0], [-1, 1]),
      xyxy.slice([0, 3], [-1, 1]), xyxy.slice([0, 2], [-1, 1])
    ]);
    const rounded_coords = tf.div(yxyx.transpose(), [h, w, h, w]);
    return tf.image.cropAndResize(image, rounded_coords, [0], crop_size);
  }

  scale(factors) {
    const starts = tf.mul(this.startPoint, factors);
    const ends = tf.mul(this.endPoint, factors);

    const new_coordinates = tf.concat2d([starts, ends], 1);
    return new Box(new_coordinates);
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const new_size = tf.mul(tf.div(size, 2), ratio);

    const new_starts = tf.sub(centers, new_size);
    const new_ends = tf.add(centers, new_size);

    return new Box(tf.concat2d([new_starts, new_ends], 1));
  }
}
