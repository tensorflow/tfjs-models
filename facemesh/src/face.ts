import * as tf from '@tensorflow/tfjs-core';

export class BlazeFaceModel {
  constructor(model, width, height) {
    this.blazeFaceModel = model;
    this.width = width;
    this.height = height;
    this.config = this._get_anchors_config();
    this.anchors = this._generate_anchors(width, height, this.config);
    this.input_size = tf.tensor([width, height]);

    this.iou_threshold = 0.3;
    this.scoreThreshold = 0.75;
  }

  _get_anchors_config() {
    return {
      'strides': [8, 16],
      'anchors': [2, 6],
    };
  };

  _generate_anchors(width, height, output_spec) {
    const anchors = [];
    for (let i = 0; i < output_spec.strides.length; ++i) {
      const stride = output_spec.strides[i];
      const grid_rows = Math.floor((height + stride - 1) / stride);
      const grid_cols = Math.floor((width + stride - 1) / stride);
      const anchors_num = output_spec.anchors[i];

      for (let grid_y = 0; grid_y < grid_rows; ++grid_y) {
        const anchor_y = stride * (grid_y + 0.5);

        for (let grid_x = 0; grid_x < grid_cols; ++grid_x) {
          const anchor_x = stride * (grid_x + 0.5);
          for (let n = 0; n < anchors_num; n++) {
            anchors.push([anchor_x, anchor_y]);
          }
        }
      }
    }
    return tf.tensor(anchors);
  }

  _decode_bounds(box_outputs) {
    const box_starts = tf.slice(box_outputs, [0, 0], [-1, 2]);
    const centers = tf.add(box_starts, this.anchors);

    const box_sizes = tf.slice(box_outputs, [0, 2], [-1, 2]);

    const box_sizes_norm = tf.div(box_sizes, this.input_size);
    const centers_norm = tf.div(centers, this.input_size);

    const starts = tf.sub(centers_norm, tf.div(box_sizes_norm, 2));
    const ends = tf.add(centers_norm, tf.div(box_sizes_norm, 2));

    return tf.concat2d(
        [tf.mul(starts, this.input_size), tf.mul(ends, this.input_size)], 1);
  }

  _getBoundingBox(input_image) {
    const img = tf.mul(tf.sub(input_image, 0.5), 2);  // make input [-1, 1]

    const detect_outputs = this.blazeFaceModel.predict(img);

    const scores =
        tf.sigmoid(tf.slice(detect_outputs, [0, 0, 0], [1, -1, 1])).squeeze();

    const box_regressors =
        tf.slice(detect_outputs, [0, 0, 1], [1, -1, 4]).squeeze();
    const boxes = this._decode_bounds(box_regressors);
    const box_indices =
        tf.image
            .nonMaxSuppression(
                boxes, scores, 1, this.iou_threshold, this.scoreThreshold)
            .arraySync();

    if (box_indices.length == 0) {
      return null;  // TODO (vakunov): don't return null. Empty box?
    }

    // TODO (vakunov): change to multi face case
    const box_index = box_indices[0];
    const result_box = tf.slice(boxes, [box_index, 0], [1, -1]);
    return result_box.arraySync();
  }

  getSingleBoundingBox(input_image) {
    const original_h = input_image.shape[1];
    const original_w = input_image.shape[2];

    const image =
        input_image.resizeBilinear([this.width, this.height]).div(255);
    const bboxes = this._getBoundingBox(image);

    if (!bboxes) {
      return null;
    }

    const factors = tf.div([original_w, original_h], this.input_size);
    return new Box(tf.tensor(bboxes)).scale(factors);
  }
}
