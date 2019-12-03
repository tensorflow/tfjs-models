const MAX_CONTINUOUS_CHECKS = 5;
const LANDMARKS_COUNT = 21;

function degToRadian(deg) {
  return deg * (Math.PI / 180);
}

function NormalizeRadians(angle) {
  return angle - 2 * Math.PI * Math.floor((angle - (-Math.PI)) / (2 * Math.PI));
}

function computeRotation(point1, point2) {
  const radians = Math.PI / 2 - Math.atan2(-(point2[1] - point1[1]),
    point2[0] - point1[0]);
  return NormalizeRadians(radians);
}


class HandPipeline {
  constructor(handdetect, handtrackModel) {
    this.handdetect = handdetect;
    this.handtrackModel = handtrackModel;
    this.runs_without_hand_detector = 0;
    this.force_update = false;
    this.max_hands_num = 1; // simple case
    this.rois = [];
  }

  /**
   * Calculates face mesh for specific image (468 points).
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

    const box = this.rois[0];

    // TODO (vakunov): move to configuration
    const width = 256., height = 256.;

    const scale_factor = 2.6;
    const shifts = [0, -0.5];

    const angle = this.calculateRotation(box);

    const box_for_cut = this.transform_box(box, angle, scale_factor, shifts);

    const cutted_hand = box_for_cut.cutFromAndResize(image_tensor, [height, width]);

    const handImage = tf.image.rotate(cutted_hand, angle).div(255);

    this.showImage(handImage);
    const output = this.handtrackModel.predict(handImage);

    const coords3d = tf.reshape(output[0], [-1, 3]);
    const coords2d = coords3d.slice([0, 0], [-1, 2]);

    const coords2d_scaled = tf.mul(coords2d,
      tf.div(box.getSize(), [width, height]))
      .add(box.startPoint);

    const landmarks_box = this.calculateLandmarsBoundingBox(coords2d_scaled);
    this.updateROIFromFacedetector(landmarks_box);

    const handFlag = output[1].arraySync()[0][0];

    console.log(handFlag);

    if (handFlag < 0.9) { // TODO: move to configuration
      this.clearROIS();
    }

    return coords2d_scaled;
  }

  transform_box(box, rotation, scale_factor, shifts) {
    const new_box = box.increaseBox(scale_factor);

    const boxSize = tf.sub(new_box.endPoint, new_box.startPoint);

    const absolute_shifts = tf.mul(boxSize, tf.tensor(shifts));

    const cosa = Math.cos(rotation);
    const sina = Math.sin(rotation);

    const rotation_matrix = tf.tensor([[cosa, -sina], [sina, cosa]]);
    const rotated_shifts = tf.matMul(rotation_matrix, tf.transpose(absolute_shifts));

    const new_start = tf.add(new_box.startPoint, rotated_shifts.transpose());
    const new_end = tf.add(new_box.endPoint, rotated_shifts.transpose());

    const new_coordinates = tf.concat2d([new_start, new_end], 1);
    return new Box(new_coordinates);

    // rect->set_x_center(rect->x_center() + x_shift);
    // rect->set_y_center(rect->y_center() + y_shift);
  }

  calculateRotation(box) {
    let keypointsArray = box.landmarks.arraySync();
    if (keypointsArray.length == 7) {
      return computeRotation(keypointsArray[0], keypointsArray[2]);
    } else if (keypointsArray.length == 21) {
      return computeRotation(keypointsArray[0], keypointsArray[9]);
    }
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
    const image = cutted_hand.squeeze([0]);// tf.div(, 255);

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

  calculateLandmarsBoundingBox(landmarks) {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const box_min_max = tf.stack([xs.min(), ys.min(), xs.max(), ys.max()]);
    return new Box(box_min_max.expandDims(0), landmarks);
  }
}
