import * as tf from '@tensorflow/tfjs-core';

export type BoxType = {
  startEndTensor: tf.Tensor,
  startPoint: tf.Tensor,
  endPoint: tf.Tensor,
  landmarks?: tf.Tensor
};

export class Box {
  public startEndTensor: tf.Tensor;
  public startPoint: tf.Tensor;
  public endPoint: tf.Tensor;
  public landmarks?: tf.Tensor;

  constructor(startEndTensor: tf.Tensor, landmarks?: tf.Tensor) {
    // keep tensor for the next frame
    this.startEndTensor = tf.keep(startEndTensor);
    // startEndTensor[:, 0:2]
    this.startPoint = tf.keep(tf.slice(startEndTensor, [0, 0], [-1, 2]));
    // startEndTensor[:, 2:4]
    this.endPoint = tf.keep(tf.slice(startEndTensor, [0, 2], [-1, 2]));
    if (landmarks) {
      this.landmarks = tf.keep(landmarks);
    }
  }

  dispose() {
    tf.dispose(this.startEndTensor);
    tf.dispose(this.startPoint);
    tf.dispose(this.endPoint);
  }

  getSize() {
    return tf.abs(tf.sub(this.endPoint, this.startPoint));
  }

  getCenter() {
    const halfSize = tf.div(tf.sub(this.endPoint, this.startPoint), 2);
    return tf.add(this.startPoint, halfSize);
  }

  cutFromAndResize(image: tf.Tensor4D, crop_size: [number, number]) {
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
    const rounded_coords =
        tf.div(yxyx.transpose(), [h, w, h, w]) as tf.Tensor2D;
    return tf.image.cropAndResize(image, rounded_coords, [0], crop_size);
  }

  scale(factors: any) {
    const starts = tf.mul(this.startPoint, factors);
    const ends = tf.mul(this.endPoint, factors);

    const new_coordinates =
        tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);
    return new Box(new_coordinates, tf.mul(this.landmarks, factors));
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const new_size = tf.mul(tf.div(size, 2), ratio);

    const new_starts = tf.sub(centers, new_size);
    const new_ends = tf.add(centers, new_size);

    return new Box(
        tf.concat2d([new_starts as tf.Tensor2D, new_ends as tf.Tensor2D], 1),
        this.landmarks);
  }
}
