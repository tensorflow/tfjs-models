import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

// import {rotate as rotateCpu} from './rotate_cpu';
import {rotate as rotateWebgl} from './rotate_gpu';

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

function NormalizeRadians(angle: number) {
  return angle - 2 * Math.PI * Math.floor((angle - (-Math.PI)) / (2 * Math.PI));
}

function computeRotation(point1: [number, number], point2: [number, number]) {
  const radians =
      Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]), point2[0] - point1[0]);
  return NormalizeRadians(radians);
}

class HandPipeline {
  private handdetect: HandDetectModel;
  private handtrackModel: tfconv.GraphModel;
  private runsWithoutHandDetector: number;
  private forceUpdate: boolean;
  private maxHandsNum: number;
  private rois: any[];

  constructor(handdetect: HandDetectModel, handtrackModel: tfconv.GraphModel) {
    this.handdetect = handdetect;
    this.handtrackModel = handtrackModel;
    this.runsWithoutHandDetector = 0;
    this.forceUpdate = false;
    this.maxHandsNum = 1;  // simple case
    this.rois = [];
  }

  calculateHandPalmCenter(box: any) {
    return tf.gather(box.landmarks, [0, 2]).mean(0);
  }

  /**
   * Calculates hand mesh for specific image (21 points).
   *
   * @param {tf.Tensor!} input - image tensor of shape [1, H, W, 3].
   *
   * @return {tf.Tensor?} tensor of 2d coordinates (1, 21, 3)
   */
  next_meshes(input: tf.Tensor3D|ImageData|HTMLVideoElement|HTMLImageElement|
              HTMLCanvasElement) {
    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });

    if (this.needROIUpdate()) {
      const box = this.handdetect.getSingleBoundingBox(image);
      if (!box) {
        this.clearROIS();
        return null;
      }
      this.updateROIFromFacedetector(box);
      this.runsWithoutHandDetector = 0;
      this.forceUpdate = false;
    } else {
      this.runsWithoutHandDetector++;
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
        [x[0] / image.shape[2], x[1] / image.shape[1]];
    const rotated_image = rotateWebgl(
        image, angle, 0, handpalm_center_relative as [number, number]);
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

    const cutted_hand = box_for_cut.cutFromAndResize(
        rotated_image as tf.Tensor4D, [width, height]);
    const handImage = cutted_hand.div(255);

    const output = this.handtrackModel.predict(handImage) as tf.Tensor[];

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

    const handFlag = ((output[1] as tf.Tensor).arraySync() as number[][])[0][0];
    if (handFlag < 0.8) {  // TODO: move to configuration
      this.clearROIS();
      return null;
    }

    return coords2d_result;
  }

  inverse(matrix: tf.Tensor) {
    const rotation_part = tf.slice(matrix, [0, 0], [2, 2]).transpose();
    const translate_part = tf.slice(matrix, [0, 2], [2, 1]);
    const change_translation = tf.neg(tf.matMul(rotation_part, translate_part));
    const inverted = tf.concat([rotation_part, change_translation], 1);
    return tf.concat([inverted, tf.tensor([[0, 0, 1]])], 0);
  }

  makeSquareBox(box: Box) {
    const centers = box.getCenter();
    const size = box.getSize();
    const maxEdge = tf.max(size, 1);

    const half_size = tf.div(maxEdge, 2);

    const new_starts = tf.sub(centers, half_size);
    const new_ends = tf.add(centers, half_size);

    return new Box(
        tf.concat2d([new_starts as tf.Tensor2D, new_ends as tf.Tensor2D], 1));
  }

  shiftBox(box: any, shifts: number[]) {
    const boxSize =
        tf.sub(box.endPoint as tf.Tensor, box.startPoint as tf.Tensor);
    const absolute_shifts = tf.mul(boxSize, tf.tensor(shifts));
    const new_start = tf.add(box.startPoint, absolute_shifts);
    const new_end = tf.add(box.endPoint, absolute_shifts);
    const new_coordinates = tf.concat2d([new_start as any, new_end], 1);

    return new Box(new_coordinates);
  }

  calculateLandmarksBoundingBox(landmarks: tf.Tensor) {
    const xs = landmarks.slice([0, 0], [-1, 1]);
    const ys = landmarks.slice([0, 1], [-1, 1]);

    const box_min_max = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    return new Box(box_min_max.expandDims(0), landmarks);
  }

  build_translation_matrix(translation: tf.Tensor) {
    // last column
    const only_tranalation =
        tf.pad(translation.expandDims(0), [[2, 0], [0, 1]]).transpose();

    return tf.add(tf.eye(3), only_tranalation);
  }

  build_rotation_matrix_with_center(rotation: number, center: tf.Tensor) {
    const cosa = Math.cos(rotation);
    const sina = Math.sin(rotation);

    const rotation_matrix =
        tf.tensor([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]]);

    return tf.matMul(
        tf.matMul(this.build_translation_matrix(center), rotation_matrix),
        this.build_translation_matrix(tf.neg(center)));
  }

  build_rotation_matrix(rotation: number, center: tf.Tensor) {
    const cosa = Math.cos(rotation);
    const sina = Math.sin(rotation);

    const rotation_matrix = tf.tensor([[cosa, -sina], [sina, cosa]]);
    return rotation_matrix;
  }


  calculateRotation(box: BoxType) {
    let keypointsArray = box.landmarks.arraySync() as [number, number][];
    return computeRotation(keypointsArray[0], keypointsArray[2]);
  }

  updateROIFromFacedetector(box: any) {
    this.rois = [box];
  }

  clearROIS() {
    for (let roi in this.rois) {
      tf.dispose(roi);
    }
    this.rois = [];
  }

  needROIUpdate() {
    const rois_count = this.rois.length;
    const has_no_rois = rois_count == 0;
    const should_check_for_more_hands = rois_count != this.maxHandsNum &&
        this.runsWithoutHandDetector >= MAX_CONTINUOUS_CHECKS;

    return this.forceUpdate || has_no_rois || should_check_for_more_hands;
  }
}

class HandDetectModel {
  private model: tfconv.GraphModel;
  private anchors: tf.Tensor;
  private input_size: tf.Tensor;
  private iou_threshold: number;
  private scoreThreshold: number;

  constructor(model: tfconv.GraphModel, width: number, height: number) {
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

  _decode_bounds(box_outputs: tf.Tensor) {
    const box_starts = tf.slice(box_outputs, [0, 0], [-1, 2]);
    const centers = tf.add(tf.div(box_starts, this.input_size), this.anchors);
    const box_sizes = tf.slice(box_outputs, [0, 2], [-1, 2]);

    const box_sizes_norm = tf.div(box_sizes, this.input_size);
    const centers_norm = centers;

    const starts = tf.sub(centers_norm, tf.div(box_sizes_norm, 2));
    const ends = tf.add(centers_norm, tf.div(box_sizes_norm, 2));

    return tf.concat2d(
        [
          tf.mul(starts as tf.Tensor2D, this.input_size as tf.Tensor2D) as
              tf.Tensor2D,
          tf.mul(ends, this.input_size) as tf.Tensor2D
        ],
        1);
  }

  _decode_landmarks(raw_landmarks: tf.Tensor) {
    const relative_landmarks = tf.add(
        tf.div(raw_landmarks.reshape([-1, 7, 2]), this.input_size),
        this.anchors.reshape([-1, 1, 2]));

    return tf.mul(relative_landmarks, this.input_size);
  }

  _getBoundingBox(input_image: tf.Tensor) {
    const img = tf.mul(tf.sub(input_image, 0.5), 2);  // make input [-1, 1]

    const detect_outputs = this.model.predict(img) as tf.Tensor;

    const scores =
        tf.sigmoid(tf.slice(detect_outputs, [0, 0, 0], [1, -1, 1])).squeeze() as
        tf.Tensor1D;

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

  getSingleBoundingBox(input_image: tf.Tensor4D) {
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

type BoxType = {
  startEndTensor: tf.Tensor,
  startPoint: tf.Tensor,
  endPoint: tf.Tensor,
  landmarks?: tf.Tensor
};

class Box {
  private startEndTensor: tf.Tensor;
  private startPoint: tf.Tensor;
  private endPoint: tf.Tensor;
  private landmarks?: tf.Tensor;

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
