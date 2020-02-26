import * as tf from '@tensorflow/tfjs-core';

export type CPUBoxType = {
  startEndTensor: [number, number, number, number],
  startPoint: [number, number],
  endPoint: [number, number],
  landmarks?: Array<[number, number]>
};

export class CPUBox {
  public startEndTensor: [number, number, number, number];
  public startPoint: [number, number];
  public endPoint: [number, number];
  public landmarks?: Array<[number, number]>;

  constructor(
      startEndTensor: [number, number, number, number],
      landmarks?: Array<[number, number]>) {
    // keep tensor for the next frame
    // this.startEndTensor = tf.keep(startEndTensor);
    this.startEndTensor = startEndTensor;
    // startEndTensor[:, 0:2]
    // this.startPoint = tf.keep(tf.slice(startEndTensor, [0, 0], [-1, 2]));
    this.startPoint = [startEndTensor[0], startEndTensor[1]];
    // startEndTensor[:, 2:4]
    // this.endPoint = tf.keep(tf.slice(startEndTensor, [0, 2], [-1, 2]));
    this.endPoint = [startEndTensor[2], startEndTensor[3]];
    if (landmarks) {
      // this.landmarks = tf.keep(landmarks);
      this.landmarks = landmarks;
    }
  }

  dispose() {
    // tf.dispose(this.startEndTensor);
    // tf.dispose(this.startPoint);
    // tf.dispose(this.endPoint);
  }

  getSize() {
    return [
      Math.abs(this.endPoint[0] - this.startPoint[0]),
      Math.abs(this.endPoint[1] - this.startPoint[1])
    ];
    // return tf.abs(tf.sub(this.endPoint, this.startPoint));
  }

  getCenter(): [number, number] {
    // const halfSize = tf.div(tf.sub(this.endPoint, this.startPoint), 2);
    // return tf.add(this.startPoint, halfSize);

    return [
      this.startPoint[0] + (this.endPoint[0] - this.startPoint[0]) / 2,
      this.startPoint[1] + (this.endPoint[1] - this.startPoint[1]) / 2
    ];
  }

  cutFromAndResize(image: tf.Tensor4D, crop_size: [number, number]) {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    // const yxyx = tf.concat2d(
    //     [
    //       xyxy.slice([0, 1], [-1, 1]) as tf.Tensor2D,
    //       xyxy.slice([0, 0], [-1, 1]) as tf.Tensor2D,
    //       xyxy.slice([0, 3], [-1, 1]) as tf.Tensor2D,
    //       xyxy.slice([0, 2], [-1, 1]) as tf.Tensor2D
    //     ],
    //     0);
    const yxyx = [xyxy[1], xyxy[0], xyxy[3], xyxy[2]];
    // const rounded_coords =
    //     tf.div(yxyx.transpose(), [h, w, h, w]) as tf.Tensor2D;
    const rounded_coords = [yxyx[0] / h, yxyx[1] / w, yxyx[2] / h, yxyx[3] / w];
    return tf.image.cropAndResize(image, [rounded_coords], [0], crop_size);
  }

  scale(factors: [number, number]) {
    // const starts = tf.mul(this.startPoint, factors);
    // const ends = tf.mul(this.endPoint, factors);

    const starts =
        [this.startPoint[0] * factors[0], this.startPoint[1] * factors[1]];
    const ends = [this.endPoint[0] * factors[0], this.endPoint[1] * factors[1]];

    // const new_coordinates =
    //     tf.concat2d([starts as tf.Tensor2D, ends as tf.Tensor2D], 1);
    // return new Box(new_coordinates, tf.mul(this.landmarks, factors));

    return new CPUBox(
        [...starts, ...ends] as [number, number, number, number],
        this.landmarks.map((coord: [number, number]) => {
          return [coord[0] * factors[0], coord[1] * factors[1]];
        }) as [number, number][]);
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    // const new_size = tf.mul(tf.div(size, 2), ratio);

    const new_size = [ratio * size[0] / 2, ratio * size[1] / 2];
    const new_starts = [centers[0] - new_size[0], centers[1] - new_size[1]];
    const new_ends = [centers[0] + new_size[0], centers[1] + new_size[1]];

    // const new_starts = tf.sub(centers, new_size);
    // const new_ends = tf.add(centers, new_size);

    // return new Box(
    //     tf.concat2d([new_starts as tf.Tensor2D, new_ends as tf.Tensor2D], 1),
    //     this.landmarks);

    return new CPUBox(
        new_starts.concat(new_ends) as [number, number, number, number],
        this.landmarks);
  }
}
