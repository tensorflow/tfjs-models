/**
 * @fileoverview Description of this file.
 */

class HandDetectModel {

  constructor(model, width, height) {
    this.model = model;
    this.anchors = this._generate_anchors();
    this.input_size = tf.tensor([width, height]);

    this.iou_threshold = 0.3;
    this.scoreThreshold = 0.5;
  }

  _generate_anchors() {
    const anchors = [];

    for (let i = 0; i < ANCHORS.length; ++i) {
      const anchor = ANCHORS[i];
      anchors.push([anchor.x_center, anchor.y_center]);
    }
    return tf.tensor(anchors);
  }

  _decode_bounds(box_outputs) {
    const box_starts = tf.slice(box_outputs, [0, 0], [-1, 2]);
    const centers = tf.add(tf.div(box_starts, this.input_size), this.anchors);
    const box_sizes = tf.slice(box_outputs, [0, 2], [-1, 2]);

    const box_sizes_norm = tf.div(box_sizes, this.input_size);
    const centers_norm = centers;

    const starts = tf.sub(centers_norm, tf.div(box_sizes_norm, 2));
    const ends = tf.add(centers_norm, tf.div(box_sizes_norm, 2));

    return tf.concat2d([tf.mul(starts, this.input_size),
    tf.mul(ends, this.input_size)], 1);
  }

  _decode_landmarks(raw_landmarks) {
    const relative_landmarks = tf.add(
      tf.div(raw_landmarks.reshape([-1, 7, 2]), this.input_size),
      this.anchors.reshape([-1, 1, 2]));

    return tf.mul(relative_landmarks, this.input_size);

  }

  _getBoundingBox(input_image) {
    const img = tf.mul(tf.sub(input_image, 0.5), 2); // make input [-1, 1]

    const detect_outputs = this.model.predict(img);

    const scores = tf.sigmoid(tf.slice(detect_outputs, [0, 0, 0], [1, -1, 1]))
      .squeeze();

    const raw_boxes = tf.slice(detect_outputs, [0, 0, 1], [1, -1, 4]).squeeze();
    const raw_landmarks = tf.slice(detect_outputs, [0, 0, 5],
      [1, -1, 14]).squeeze();
    const boxes = this._decode_bounds(raw_boxes);

    const box_indices = tf.image.nonMaxSuppression(boxes, scores, 1,
      this.iou_threshold, this.scoreThreshold).arraySync();

    const landmarks = this._decode_landmarks(raw_landmarks);
    if (box_indices.length == 0) {
      return [null, null]; // TODO (vakunov): don't return null. Empty box?
    }

    // TODO (vakunov): change to multi face case
    const box_index = box_indices[0];
    const result_box = tf.slice(boxes, [box_index, 0], [1, -1]);

    const result_landmarks = tf.slice(landmarks, [box_index, 0], [1])
      .reshape([-1, 2]);
    return [result_box, result_landmarks];
  }
  // landmarks.print();

  getSingleBoundingBox(input_image) {
    const original_h = input_image.shape[1];
    const original_w = input_image.shape[2];

    const image = input_image.resizeBilinear([256, 256]).div(255);
    const bboxes_data = this._getBoundingBox(image);

    if (!bboxes_data[0]) {
      return null;
    }

    const bboxes = bboxes_data[0].arraySync();
    const landmarks = bboxes_data[1];

    const factors = tf.div([original_w, original_h], this.input_size);
    return new Box(tf.tensor(bboxes), landmarks).scale(factors);
  }
};

class Box {
  constructor(startEndTensor, landmarks) {
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

  cutFromAndResize(image, crop_size) {
    const h = image.shape[1];
    const w = image.shape[2];

    const xyxy = this.startEndTensor;
    const yxyx = tf.concat2d([xyxy.slice([0, 1], [-1, 1]),
    xyxy.slice([0, 0], [-1, 1]),
    xyxy.slice([0, 3], [-1, 1]),
    xyxy.slice([0, 2], [-1, 1])]);
    const rounded_coords = tf.div(yxyx.transpose(), [h, w, h, w]);
    return tf.image.cropAndResize(image, rounded_coords, [0], crop_size);
  }

  scale(factors) {
    const starts = tf.mul(this.startPoint, factors);
    const ends = tf.mul(this.endPoint, factors);

    const new_coordinates = tf.concat2d([starts, ends], 1);
    return new Box(new_coordinates, tf.mul(this.landmarks, factors));
  }

  increaseBox(ratio = 1.5) {
    const centers = this.getCenter();
    const size = this.getSize();

    const new_size = tf.mul(tf.div(size, 2), ratio);

    const new_starts = tf.sub(centers, new_size);
    const new_ends = tf.add(centers, new_size);

    return new Box(tf.concat2d([new_starts, new_ends], 1), this.landmarks);
  }
}
