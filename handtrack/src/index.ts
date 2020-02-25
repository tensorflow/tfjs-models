import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import {Box, BoxType} from './box';
import {HandDetectModel} from './hand';
import {maxPool, maxPoolWArgMax} from './pool_gpu';
import {rotate as rotateWebgl} from './rotate_gpu';

// const HANDDETECT_MODEL_PATH =
//     'https://storage.googleapis.com/learnjs-data/tfjs_converter_v1.3.2_master/handdetector_hourglass_short_2019_03_25_v0/model.json';
const HANDDETECT_MODEL_PATH =
    'https://tfhub.dev/tensorflow/tfjs-model/handdetector/1/default/1';

// NEW MODEL
// const HANDTRACK_MODEL_PATH =
//     'https://storage.googleapis.com/learnjs-data/handskeleton_handflag_light_2019_12_18_v0.hdf5_tfjs/model.json';
// const HANDTRACK_MODEL_PATH =
//     'https://storage.googleapis.com/learnjs-data/handskeleton_handflag_2019_12_18_v0.hdf5_tfjs/model.json';

// OLD MODEL
// const HANDTRACK_MODEL_PATH =
//     'https://storage.googleapis.com/learnjs-data/tfjs_converter_v1.3.2_master/handskeleton_3d_handflag_2019_08_19_v0/model.json';
const HANDTRACK_MODEL_PATH =
    'https://tfhub.dev/tensorflow/tfjs-model/handskeleton/1/default/1';

tfconv.registerOp('MaxPoolWithArgmax', (obj: any) => {
  const input = obj['inputs'][0];
  const attrs = obj['attrs'];
  const maxVals = maxPool(input, attrs);
  const argmax = maxPoolWArgMax(input, attrs);
  return [maxVals as tf.Tensor, argmax as tf.Tensor];
});

export async function load() {
  const ANCHORS =
      await tf.util
          .fetch(
              'https://storage.googleapis.com/learnjs-data/handtrack_staging/anchors.json')
          .then(d => d.json());

  const handModel =
      await tfconv.loadGraphModel(HANDDETECT_MODEL_PATH, {fromTFHub: true});
  const handSkeletonModel =
      await tfconv.loadGraphModel(HANDTRACK_MODEL_PATH, {fromTFHub: true});

  const handdetect = new HandDetectModel(handModel, 256, 256, ANCHORS);
  const pipeline = new HandPipeline(handdetect, handSkeletonModel);

  return pipeline;
}

const MAX_CONTINUOUS_CHECKS = 100;
const BRANCH_ON_DETECTION = true;  // whether we branch box scaling / shifting
                                   // logic depending on detection type

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
  private maxHandsNum: number;
  private rois: any[];

  constructor(handdetect: HandDetectModel, handtrackModel: tfconv.GraphModel) {
    this.handdetect = handdetect;
    this.handtrackModel = handtrackModel;
    this.runsWithoutHandDetector = 0;
    this.maxHandsNum = 1;  // simple case
    this.rois = [];
  }

  calculateHandPalmCenter(box: any) {  // DetectionsToRectsCalculator
    return tf.gather(box.landmarks, [0, 2]).mean(0);
  }

  rotateVector(angle: number, vector: [number, number]): [number, number] {
    return [
      vector[0] * Math.cos(angle) - vector[1] * Math.sin(angle),
      vector[0] * Math.sin(angle) + vector[1] * Math.cos(angle)
    ];
  }

  /**
   * Calculates hand mesh for specific image (21 points).
   *
   * @param {tf.Tensor!} input - image tensor of shape [1, H, W, 3].
   *
   * @return {tf.Tensor?} tensor of 2d coordinates (1, 21, 3)
   */
  async estimateHand(input: tf.Tensor3D|ImageData|HTMLVideoElement|
                     HTMLImageElement|HTMLCanvasElement) {
    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });

    const useFreshBox = this.needROIUpdate();

    if (useFreshBox) {
      const box = this.handdetect.getSingleBoundingBox(image);
      if (!box) {
        this.clearROIS();
        return null;
      }
      this.updateROIFromFacedetector(box, true);
      this.runsWithoutHandDetector = 0;
    } else {
      this.runsWithoutHandDetector++;
    }

    const scaledCoords = tf.tidy(() => {
      const width = 256., height = 256.;
      const box = this.rois[0];

      const angle = this.calculateRotation(box);

      const handpalm_center = box.getCenter().gather(0);
      const x = handpalm_center.arraySync();
      const handpalm_center_relative =
          [x[0] / image.shape[2], x[1] / image.shape[1]];
      const rotated_image = rotateWebgl(
          image, angle, 0, handpalm_center_relative as [number, number]);
      const palm_rotation_matrix =
          this.build_rotation_matrix_with_center(-angle, handpalm_center);

      let box_for_cut, bbRotated, bbShifted, bbSquarified;
      if (!BRANCH_ON_DETECTION || useFreshBox) {
        const numLandmarks = box.landmarks.shape[0];
        const box_landmarks_homo = tf.concat(
            [box.landmarks, tf.ones([numLandmarks]).expandDims(1)], 1);
        const rotated_landmarks =
            tf.matMul(box_landmarks_homo, palm_rotation_matrix, false, true)
                .slice([0, 0], [numLandmarks, 2]);

        bbRotated = this.calculateLandmarksBoundingBox(rotated_landmarks);
        const shiftVector: [number, number] = [0, -0.4];
        const rotatedVector = this.rotateVector(angle, shiftVector);
        bbShifted = this.shiftBox(bbRotated, rotatedVector);
        bbSquarified = this.makeSquareBox(bbShifted);
        box_for_cut = bbSquarified.increaseBox(3.0);
      } else {
        box_for_cut = box;
      }

      const cutted_hand = box_for_cut.cutFromAndResize(
          rotated_image as tf.Tensor4D, [width, height]);
      const handImage = cutted_hand.div(255);

      const output = this.handtrackModel.predict(handImage) as tf.Tensor[];

      const output_keypoints = output[output.length - 1];
      const coords3d = tf.reshape(output_keypoints, [-1, 3]);
      const coords2d = coords3d.slice([0, 0], [-1, 2]);

      // center around 0, 0, scale to fit 256, 256
      const coords2d_scaled = tf.mul(
          coords2d.sub(tf.tensor([128, 128])),
          tf.div(box_for_cut.getSize(), [width, height]));

      const coords_rotation_matrix =
          this.build_rotation_matrix_with_center(angle, tf.tensor([0, 0]));

      const coords2d_homo = tf.concat(
          [
            coords2d_scaled,
            tf.ones([output_keypoints.shape[1] / 3]).expandDims(1)
          ],
          1);

      const coords2d_rotated =
          tf.matMul(coords2d_homo, coords_rotation_matrix, false, true)
              .slice([0, 0], [-1, 2]);

      const original_center =
          tf.matMul(
                tf.concat(
                    [box_for_cut.getCenter(), tf.ones([1]).expandDims(1)], 1),
                this.inverse(palm_rotation_matrix), false, true)
              .slice([0, 0], [1, 2]);

      const coords2d_result = coords2d_rotated.add(original_center);

      const landmarks_ids = [0, 5, 9, 13, 17, 1, 2];
      const selected_landmarks = tf.gather(coords2d_result, landmarks_ids);

      let nextBoundingBox;
      if (BRANCH_ON_DETECTION) {
        const landmarks_box =
            this.calculateLandmarksBoundingBox(coords2d_result);

        const landmarks_box_shifted =
            this.shiftBox(landmarks_box, this.rotateVector(angle, [0, -0.05]));
        const landmarks_box_shifted_squarified =
            this.makeSquareBox(landmarks_box_shifted);
        nextBoundingBox = landmarks_box_shifted_squarified.increaseBox(1.75);
        (nextBoundingBox as any).landmarks = tf.keep(selected_landmarks);
      } else {
        nextBoundingBox =
            this.calculateLandmarksBoundingBox(selected_landmarks);
      }

      this.updateROIFromFacedetector(
          nextBoundingBox as any, false /* force replace */);

      const handFlag =
          ((output[0] as tf.Tensor).arraySync() as number[][])[0][0];
      if (handFlag < 0.8) {  // TODO: move to configuration
        this.clearROIS();
        return null;
      }

      let result = [coords2d_result];
      if (location.hash === '#debug') {
        result = result.concat([
          angle, cutted_hand, box as any, bbRotated as any, bbShifted as any,
          bbSquarified as any, nextBoundingBox as any
        ]);
      }

      return result;
    });

    image.dispose();
    return scaledCoords;
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

  updateROIFromFacedetector(box: any, force: boolean) {
    if (force) {
      this.rois = [box];
    } else {
      const prev = this.rois[0];
      let iou = 0;

      if (prev && prev.startPoint) {
        const boxStartEnd = box.startEndTensor.arraySync()[0];
        const prevStartEnd = prev.startEndTensor.arraySync()[0];

        const xBox = Math.max(boxStartEnd[0], prevStartEnd[0]);
        const yBox = Math.max(boxStartEnd[1], prevStartEnd[1]);
        const xPrev = Math.min(boxStartEnd[2], prevStartEnd[2]);
        const yPrev = Math.min(boxStartEnd[3], prevStartEnd[3]);

        const interArea = (xPrev - xBox) * (yPrev - yBox);

        const boxArea = (boxStartEnd[2] - boxStartEnd[0]) *
            (boxStartEnd[3] - boxStartEnd[1]);
        const prevArea = (prevStartEnd[2] - prevStartEnd[0]) *
            (prevStartEnd[3] - boxStartEnd[1]);
        iou = interArea / (boxArea + prevArea - interArea);
      }

      if (iou > 0.8) {
        this.rois[0] = prev;
      } else {
        this.rois[0] = box;
      }
    }
  }

  clearROIS() {
    for (let roi in this.rois) {
      tf.dispose(roi);
    }
    this.rois = [];
  }

  needROIUpdate() {
    const rois_count = this.rois.length;

    return rois_count !== this.maxHandsNum ||
        this.runsWithoutHandDetector >= MAX_CONTINUOUS_CHECKS;
  }
}
