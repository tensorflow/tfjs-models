import * as tf from '@tensorflow/tfjs-core';
import * as tfl from '@tensorflow/tfjs-layers';

import {Box} from './box';

export class BlazeFaceModel {
  private blazeFaceModel: tfl.LayersModel;
  private width: number;
  private height: number;
  private config: any;
  private anchors: any;
  private inputSize: tf.Tensor;
  private iouThreshold: number;
  private scoreThreshold: number;

  constructor(model: tfl.LayersModel, width: number, height: number) {
    this.blazeFaceModel = model;
    this.width = width;
    this.height = height;
    this.config = this._get_anchors_config();
    this.anchors = this._generate_anchors(width, height, this.config);
    this.inputSize = tf.tensor([width, height]);

    this.iouThreshold = 0.3;
    this.scoreThreshold = 0.75;
  }

  _get_anchors_config() {
    return {
      'strides': [8, 16],
      'anchors': [2, 6],
    };
  };

  _generate_anchors(width: number, height: number, outputSpec: any) {
    const anchors = [];
    for (let i = 0; i < outputSpec.strides.length; ++i) {
      const stride = outputSpec.strides[i];
      const gridRows = Math.floor((height + stride - 1) / stride);
      const gridCols = Math.floor((width + stride - 1) / stride);
      const anchorsNum = outputSpec.anchors[i];

      for (let gridY = 0; gridY < gridRows; ++gridY) {
        const anchorY = stride * (gridY + 0.5);

        for (let gridX = 0; gridX < gridCols; ++gridX) {
          const anchorX = stride * (gridX + 0.5);
          for (let n = 0; n < anchorsNum; n++) {
            anchors.push([anchorX, anchorY]);
          }
        }
      }
    }
    return tf.tensor(anchors);
  }

  _decode_bounds(boxOutputs: tf.Tensor) {
    const box_starts = tf.slice(boxOutputs, [0, 0], [-1, 2]);
    const centers = tf.add(box_starts, this.anchors);

    const box_sizes = tf.slice(boxOutputs, [0, 2], [-1, 2]);

    const box_sizes_norm = tf.div(box_sizes, this.inputSize);
    const centers_norm = tf.div(centers, this.inputSize);

    const starts = tf.sub(centers_norm, tf.div(box_sizes_norm, 2));
    const ends = tf.add(centers_norm, tf.div(box_sizes_norm, 2));

    return tf.concat2d(
        [
          tf.mul(starts, this.inputSize) as tf.Tensor2D,
          tf.mul(ends, this.inputSize) as tf.Tensor2D
        ],
        1);
  }

  _getBoundingBox(inputImage: tf.Tensor) {
    const img = tf.mul(tf.sub(inputImage, 0.5), 2);  // make input [-1, 1]

    const detectOutputs = this.blazeFaceModel.predict(img);

    const scores =
        tf.sigmoid(
              tf.slice(detectOutputs as tf.Tensor3D, [0, 0, 0], [1, -1, 1]))
            .squeeze();

    const box_regressors =
        tf.slice(detectOutputs as tf.Tensor3D, [0, 0, 1], [1, -1, 4]).squeeze();
    const boxes = this._decode_bounds(box_regressors);
    const box_indices = tf.image
                            .nonMaxSuppression(
                                boxes, scores as tf.Tensor1D, 1,
                                this.iouThreshold, this.scoreThreshold)
                            .arraySync();

    if (box_indices.length == 0) {
      return null;  // TODO (vakunov): don't return null. Empty box?
    }

    // TODO (vakunov): change to multi face case
    const box_index = box_indices[0];
    const result_box = tf.slice(boxes, [box_index, 0], [1, -1]);
    return result_box.arraySync();
  }

  getSingleBoundingBox(inputImage: tf.Tensor4D) {
    const original_h = inputImage.shape[1];
    const original_w = inputImage.shape[2];

    const image = inputImage.resizeBilinear([this.width, this.height]).div(255);
    const bboxes = this._getBoundingBox(image);

    if (!bboxes) {
      return null;
    }

    const factors = tf.div([original_w, original_h], this.inputSize);
    return new Box(tf.tensor(bboxes)).scale(factors);
  }
}
