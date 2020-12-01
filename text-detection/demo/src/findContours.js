/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const findContours = (points, height = 100, width = 100) => {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const graph = (new Array(height * width)).fill(0);
  for (const point of points) {
    const {x, y} = point;
    graph[y * width + x] = 1;
  }
  const graphMatrix = cv.matFromArray(
      height,
      width,
      cv.CV_8U,
      graph,
  );
  cv.findContours(
      graphMatrix,
      contours,
      hierarchy,
      cv.RETR_CCOMP,
      cv.CHAIN_APPROX_SIMPLE,
  );
  const contour = contours.get(0);
  const contourData = contour.data32S;
  const contourPoints = [];

  for (let rowIdx = 0; rowIdx < contour.rows; ++rowIdx) {
    contourPoints.push(
        {x: contourData[rowIdx * 2], y: contourData[rowIdx * 2 + 1]});
  }

  graphMatrix.delete();
  hierarchy.delete();
  contours.delete();
  contour.delete();
  return contourPoints;
};
