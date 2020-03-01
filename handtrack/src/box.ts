/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// The hand bounding box.
// export type Box = {
//   startPoint: [number, number],
//   endPoint: [number, number],
//   landmarks?: Array<[number, number]>
// };

export class Box {
  public startPoint: [number, number];
  public endPoint: [number, number];
  public landmarks?: Array<[number, number]>;

  constructor(
      startPoint: [number, number], endPoint: [number, number],
      landmarks?: Array<[number, number]>) {
    this.startPoint = startPoint;
    this.endPoint = endPoint;
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

    const xyxy = this.startPoint.concat(this.endPoint);
    const yxyx = [xyxy[1], xyxy[0], xyxy[3], xyxy[2]];
    const rounded_coords = [yxyx[0] / h, yxyx[1] / w, yxyx[2] / h, yxyx[3] / w];
    return tf.image.cropAndResize(image, [rounded_coords], [0], crop_size);
  }

  scale(factors: [number, number]) {
    const starts: [number, number] =
        [this.startPoint[0] * factors[0], this.startPoint[1] * factors[1]];
    const ends: [number, number] =
        [this.endPoint[0] * factors[0], this.endPoint[1] * factors[1]];
    const newLandmarks: Array<[number, number]> = this.landmarks.map((coord: [
                                                                       number,
                                                                       number
                                                                     ]) => {
      return [coord[0] * factors[0], coord[1] * factors[1]] as [number, number];
    });

    return new Box(starts, ends, newLandmarks);
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const new_size = [ratio * size[0] / 2, ratio * size[1] / 2];
    const new_starts: [number, number] =
        [centers[0] - new_size[0], centers[1] - new_size[1]];
    const new_ends: [number, number] =
        [centers[0] + new_size[0], centers[1] + new_size[1]];

    return new Box(new_starts, new_ends, this.landmarks);
  }
}
