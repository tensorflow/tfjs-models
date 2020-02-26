import * as tf from '@tensorflow/tfjs-core';

export type BoxType = {
  startEndTensor: [number, number, number, number],
  startPoint: [number, number],
  endPoint: [number, number],
  landmarks?: Array<[number, number]>
};

export class Box {
  public startEndTensor: [number, number, number, number];
  public startPoint: [number, number];
  public endPoint: [number, number];
  public landmarks?: Array<[number, number]>;

  constructor(
      startEndTensor: [number, number, number, number],
      landmarks?: Array<[number, number]>) {
    this.startEndTensor = startEndTensor;
    this.startPoint = [startEndTensor[0], startEndTensor[1]];
    this.endPoint = [startEndTensor[2], startEndTensor[3]];
    if (landmarks) {
      this.landmarks = landmarks;
    }
  }

  getSize() {
    return [
      Math.abs(this.endPoint[0] - this.startPoint[0]),
      Math.abs(this.endPoint[1] - this.startPoint[1])
    ];
  }

  getCenter(): [number, number] {
    return [
      this.startPoint[0] + (this.endPoint[0] - this.startPoint[0]) / 2,
      this.startPoint[1] + (this.endPoint[1] - this.startPoint[1]) / 2
    ];
  }

  cutFromAndResize(image: tf.Tensor4D, crop_size: [number, number]) {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    const yxyx = [xyxy[1], xyxy[0], xyxy[3], xyxy[2]];
    const rounded_coords = [yxyx[0] / h, yxyx[1] / w, yxyx[2] / h, yxyx[3] / w];
    return tf.image.cropAndResize(image, [rounded_coords], [0], crop_size);
  }

  scale(factors: [number, number]) {
    const starts =
        [this.startPoint[0] * factors[0], this.startPoint[1] * factors[1]];
    const ends = [this.endPoint[0] * factors[0], this.endPoint[1] * factors[1]];

    return new Box(
        [...starts, ...ends] as [number, number, number, number],
        this.landmarks.map((coord: [number, number]) => {
          return [coord[0] * factors[0], coord[1] * factors[1]];
        }) as [number, number][]);
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const new_size = [ratio * size[0] / 2, ratio * size[1] / 2];
    const new_starts = [centers[0] - new_size[0], centers[1] - new_size[1]];
    const new_ends = [centers[0] + new_size[0], centers[1] + new_size[1]];

    return new Box(
        new_starts.concat(new_ends) as [number, number, number, number],
        this.landmarks);
  }
}
