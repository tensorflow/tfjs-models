/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

const MAX_CONTINUOUS_CHECKS = 5;

export class BlazePipeline {
  constructor(blazeface, blazemesh_model) {
    this.twoFiftyFive = tf.scalar(255);
    this.blazeface = blazeface;
    this.blazemesh_model = blazemesh_model;
    this.runs_without_face_detector = 0;
    this.force_update = false;
    this.max_faces_num = 1;  // simple case
    this.rois = [];
  }

  next_meshes(image_tensor) {
    const original_h = image_tensor.shape[0];
    const original_w = image_tensor.shape[1];

    // alert(["image_tensor", image_tensor]);
    const image = image_tensor.resizeBilinear([256, 256])
                      .div(this.twoFiftyFive)
                      .expandDims(0);

    if (this.needROIUpdate()) {
      const box = this.blazeface.getSingleBoundingBox(image).increaseBox();
      this.updateROIFromFacedetector(box, landmarks_filter);

      // alert(["needROIUpdate", box.startX, box.startY, box.endX, box.endY]);
      this.runs_without_face_detector = 0;
      this.force_update = false;
      console.log('Run face detector');
    } else {
      this.runs_without_face_detector++;
    }

    const box = this.rois[0];
    // console.log(box);
    const w = box.getWidth(), h = box.getHeight();
    const cutted_face = box.cutFrom(image);

    const facemesh_width = 192., facemesh_height = 192.;
    const face =
        tf.image.resizeBilinear(cutted_face, [facemesh_height, facemesh_width]);

    const output = this.blazemesh_model.predict(face);
    const coords3d = tf.reshape(output[0], [LANDMARKS_COUNT, 3]);
    const coords2d = coords3d.slice([0, 0], [LANDMARKS_COUNT, 2]);

    const coords2d_scaled = tf.mul(coords2d, [
                                w / facemesh_width, h / facemesh_height
                              ]).add(tf.tensor1d([box.startX, box.startY]));

    const landmarks_box = this.calculateLandmarsBoundingBox(coords2d_scaled);
    // const m = this.meanBox(box, landmarks_box);
    this.updateROIFromFacedetector(landmarks_box);
    // this.updateROIFromFacedetector(m);

    if (output[2].get(0, 0) < 0.9) {
      this.clearROIS();
    }

    const coords2ds = coords2d_scaled.mul(
        tf.tensor1d([original_w / 256., original_h / 256.]));

    return {coords2ds: coords2ds, cutted_face: face};
  }

  updateROIFromFacedetector(box, landmarks_filter) {
    this.rois = [box];
  }

  clearROIS() {
    this.rois = [];
  }

  meanBox(box1, box2) {
    return new Box(
        (box1.startX + box2.startX) / 2, (box1.startY + box2.startY) / 2,
        (box1.endX + box2.endX) / 2, (box1.endY + box2.endY) / 2);
  }

  needROIUpdate() {
    const rois_count = this.rois.length;
    const has_no_rois = rois_count == 0;
    const should_check_for_more_faces = rois_count != this.max_faces_num &&
        this.runs_without_face_detector >= MAX_CONTINUOUS_CHECKS;
    if (rois_count > 0) {
      const box = this.rois[0];
      if (isNaN(box.startX) || isNaN(box.startY) || isNaN(box.endX) ||
          isNaN(box.endY)) {
        return true;
      }
    }
    return this.force_update || has_no_rois || should_check_for_more_faces;
  }

  calculateLandmarsBoundingBox(landmarks) {
    const xs = landmarks.slice([0, 0], [LANDMARKS_COUNT, 1]);
    const ys = landmarks.slice([0, 1], [LANDMARKS_COUNT, 1]);

    const box =
        new Box(xs.min().get(), ys.min().get(), xs.max().get(), ys.max().get());

    const result = box.increaseBox(1.5);
    return result;
  }
}
