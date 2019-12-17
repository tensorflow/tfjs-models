import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

let ANCHORS: any;

tfconv.registerOp('Prelu', (node) => {
  const x = node.inputs[0];
  const alpha = node.inputs[1];
  return tf.prelu(x, alpha);
});

// Model loading
const HANDDETECT_MODEL_PATH =
    'https://storage.googleapis.com/learnjs-data/tfjs_converter_v1.3.2_master/handdetector_hourglass_short_2019_03_25_v0/model.json';
const HANDTRACK_MODEL_PATH =
    'https://storage.googleapis.com/learnjs-data/tfjs_converter_v1.3.2_master/handskeleton_3d_handflag_2019_08_19_v0/model.json';

export async function load() {
  // TODO: Move ANCHORS file to tfjs-models.
  ANCHORS =
      await fetch(
          'https://storage.googleapis.com/learnjs-data/handtrack_staging/anchors.json')
          .then(d => d.json());

  const handModel = await tfconv.loadGraphModel(HANDDETECT_MODEL_PATH);
  const handSkeletonModel = await tfconv.loadGraphModel(HANDTRACK_MODEL_PATH);

  const handdetect = new HandDetectModel(handModel, 256, 256);
  const pipeline = new HandPipeline(handdetect, handSkeletonModel);

  return pipeline;
}

const MAX_CONTINUOUS_CHECKS = 5;

function degToRadian(deg) {
  return deg * (Math.PI / 180);
}

function NormalizeRadians(angle) {
  return angle - 2 * Math.PI * Math.floor((angle - (-Math.PI)) / (2 * Math.PI));
}

function computeRotation(point1, point2) {
  const radians =
      Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
  return NormalizeRadians(radians);
}

class HandPipeline {
  constructor(handdetect, handtrackModel) {
    this.handdetect = handdetect;
    this.handtrackModel = handtrackModel;
    this.runs_without_hand_detector = 0;
    this.force_update = false;
    this.max_hands_num = 1;  // simple case
    this.rois = [];

    this.hand_canvas = document.getElementById('hand_cut');
    this.hand_canvas.width = 256;
    this.hand_canvas.height = 256;
    this.ctx = this.hand_canvas.getContext('2d');

    this.ctx.strokeStyle = 'red';
    // this.ctx.translate(this.hand_canvas.width, 0);
    // this.ctx.scale(-1, 1);
  }

  calculateHandPalmCenter(box) {
    return tf.gather(box.landmarks, [0, 2]).mean(0);
  }

  /**
   * Calculates hand mesh for specific image (21 points).
   *
   * @param {tf.Tensor!} image_tensor - image tensor of shape [1, H, W, 3].
   *
   * @return {tf.Tensor?} tensor of 2d coordinates (1, 21, 3)
   */
  next_meshes(image_tensor) {
    if (this.needROIUpdate()) {
      const box = this.handdetect.getSingleBoundingBox(image_tensor);
      if (!box) {
        this.clearROIS();
        return null;
      }
      this.updateROIFromFacedetector(box);
      this.runs_without_hand_detector = 0;
      this.force_update = false;
    } else {
      this.runs_without_hand_detector++;
    }

    const width = 256., height = 256.;
    const box = this.rois[0];

    // TODO (vakunov): move to configuration
    const scale_factor = 2.6;
    const shifts = [0, -0.2];
    const angle = this.calculateRotation(box);

    const handpalm_center = box.getCenter().gather(0);
    const x = handpalm_center.arraySync();
    const handpalm_center_relative =
        [x[0] / image_tensor.shape[2], x[1] / image_tensor.shape[1]];
    const rotated_image =
        tf.image.rotate(image_tensor, angle, 0, handpalm_center_relative);
    const box_landmarks_homo =
        tf.concat([box.landmarks, tf.ones([7]).expandDims(1)], 1);

    const palm_rotation_matrix =
        this.build_rotation_matrix_with_center(-angle, handpalm_center);

    const rotated_landmarks =
        tf.matMul(palm_rotation_matrix, box_landmarks_homo.transpose())
            .transpose()
            .slice([0, 0], [7, 2]);

    let box_for_cut = this.calculateLandmarksBoundingBox(rotated_landmarks)
                          .increaseBox(scale_factor);
    box_for_cut = this.makeSquareBox(box_for_cut);
    box_for_cut = this.shiftBox(box_for_cut, shifts);

    const cutted_hand =
        box_for_cut.cutFromAndResize(rotated_image, [width, height]);
    const handImage = cutted_hand.div(255);

    const output = this.handtrackModel.predict(handImage);

    const coords3d = tf.reshape(output[0], [-1, 3]);
    const coords2d = coords3d.slice([0, 0], [-1, 2]);

    const coords2d_scaled = tf.mul(
        coords2d.sub(tf.tensor([128, 128])),
        tf.div(box_for_cut.getSize(), [width, height]));

    const coords_rotation_matrix =
        this.build_rotation_matrix_with_center(angle, tf.tensor([0, 0]));

    const coords2d_homo =
        tf.concat([coords2d_scaled, tf.ones([21]).expandDims(1)], 1);

    const coords2d_rotated =
        tf.matMul(coords_rotation_matrix, coords2d_homo, false, true)
            .transpose()
            .slice([0, 0], [-1, 2]);

    const original_center =
        tf.matMul(
              this.inverse(palm_rotation_matrix),
              tf.concat(
                    [box_for_cut.getCenter(), tf.ones([1]).expandDims(1)], 1)
                  .transpose())
            .transpose()
            .slice([0, 0], [1, 2]);

    const coords2d_result = coords2d_rotated.add(original_center);

    const landmarks_ids = [0, 5, 9, 13, 17, 1, 2];
    const selected_landmarks = tf.gather(coords2d_result, landmarks_ids);

    const landmarks_box =
        this.calculateLandmarksBoundingBox(selected_landmarks);
    this.updateROIFromFacedetector(landmarks_box);

    const handFlag = output[1].arraySync()[0][0];
    if (handFlag < 0.8) {  // TODO: move to configuration
      this.clearROIS();
      return null;
    }

    return coords2d_result;
  }

  inverse(matrix) {
    const rotation_part = tf.slice(matrix, [0, 0], [2, 2]).transpose();
    const translate_part = tf.slice(matrix, [0, 2], [2, 1]);
    const change_translation = tf.neg(tf.matMul(rotation_part, translate_part));
    const inverted = tf.concat([rotation_part, change_translation], 1);
    return tf.concat([inverted, tf.tensor([[0, 0, 1]])], 0);
  }

  makeSquareBox(box) {
    const centers = box.getCenter();
    const size = box.getSize();
    const maxEdge = tf.max(size, 1);

    const half_size = tf.div(maxEdge, 2);

    const new_starts = tf.sub(centers, half_size);
    const new_ends = tf.add(centers, half_size);

    return new Box(tf.concat2d([new_starts, new_ends], 1));
  }

  shiftBox(box, shifts) {
    const boxSize = tf.sub(box.endPoint, box.startPoint);
    const absolute_shifts = tf.mul(boxSize, tf.tensor(shifts));
    const new_start = tf.add(box.startPoint, absolute_shifts);
    const new_end = tf.add(box.endPoint, absolute_shifts);
    const new_coordinates = tf.concat2d([new_start, new_end], 1);

    return new Box(new_coordinates);
  }

  calculateLandmarksBoundingBox(landmarks) {
    const xs = landmarks.slice([0, 0], [-1, 1]);
    const ys = landmarks.slice([0, 1], [-1, 1]);

    const box_min_max = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    return new Box(box_min_max.expandDims(0), landmarks);
  }

  build_translation_matrix(translation) {
    // last column
    const only_tranalation =
        tf.pad(translation.expandDims(0), [[2, 0], [0, 1]]).transpose();

    return tf.add(tf.eye(3), only_tranalation);
  }

  build_rotation_matrix_with_center(rotation, center) {
    const cosa = Math.cos(rotation);
    const sina = Math.sin(rotation);

    const rotation_matrix =
        tf.tensor([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]]);

    return tf.matMul(
        tf.matMul(this.build_translation_matrix(center), rotation_matrix),
        this.build_translation_matrix(tf.neg(center)));
  }

  build_rotation_matrix(rotation, center) {
    const cosa = Math.cos(rotation);
    const sina = Math.sin(rotation);

    const rotation_matrix = tf.tensor([[cosa, -sina], [sina, cosa]]);
    return rotation_matrix;
  }


  calculateRotation(box) {
    let keypointsArray = box.landmarks.arraySync();
    return computeRotation(keypointsArray[0], keypointsArray[2]);
  }

  updateROIFromFacedetector(box) {
    this.rois = [box];
  }

  clearROIS() {
    for (let roi in this.rois) {
      tf.dispose(roi);
    }
    this.rois = [];
  }

  showImage(cutted_hand) {
    const hand_canvas = document.getElementById('hand_cut');
    const image = cutted_hand.squeeze([0]);

    tf.browser.toPixels(tf.keep(image), hand_canvas).then((successMessage) => {
      tf.dispose(image);
    });
  }

  needROIUpdate() {
    const rois_count = this.rois.length;
    const has_no_rois = rois_count == 0;
    const should_check_for_more_hands = rois_count != this.max_hands_num &&
        this.runs_without_hand_detector >= MAX_CONTINUOUS_CHECKS;

    return this.force_update || has_no_rois || should_check_for_more_hands;
  }
}

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

    return tf.concat2d(
        [tf.mul(starts, this.input_size), tf.mul(ends, this.input_size)], 1);
  }

  _decode_landmarks(raw_landmarks) {
    const relative_landmarks = tf.add(
        tf.div(raw_landmarks.reshape([-1, 7, 2]), this.input_size),
        this.anchors.reshape([-1, 1, 2]));

    return tf.mul(relative_landmarks, this.input_size);
  }

  _getBoundingBox(input_image) {
    const img = tf.mul(tf.sub(input_image, 0.5), 2);  // make input [-1, 1]

    const detect_outputs = this.model.predict(img);

    const scores =
        tf.sigmoid(tf.slice(detect_outputs, [0, 0, 0], [1, -1, 1])).squeeze();

    const raw_boxes = tf.slice(detect_outputs, [0, 0, 1], [1, -1, 4]).squeeze();
    const raw_landmarks =
        tf.slice(detect_outputs, [0, 0, 5], [1, -1, 14]).squeeze();
    const boxes = this._decode_bounds(raw_boxes);

    const box_indices =
        tf.image
            .nonMaxSuppression(
                boxes, scores, 1, this.iou_threshold, this.scoreThreshold)
            .arraySync();

    const landmarks = this._decode_landmarks(raw_landmarks);
    if (box_indices.length == 0) {
      return [null, null];  // TODO (vakunov): don't return null. Empty box?
    }

    // TODO (vakunov): change to multi hand case
    const box_index = box_indices[0];
    const result_box = tf.slice(boxes, [box_index, 0], [1, -1]);

    const result_landmarks =
        tf.slice(landmarks, [box_index, 0], [1]).reshape([-1, 2]);
    return [result_box, result_landmarks];
  }

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
