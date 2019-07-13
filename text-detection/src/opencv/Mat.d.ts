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

// MIT License
// Copyright (c) 2017 Vincent Mühler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {Contour} from './Contour.d';
import {Moments} from './Moments.d';
import {Point2} from './Point2.d';
import {Point3} from './Point3.d';
import {Rect} from './Rect.d';
import {RotatedRect} from './RotatedRect.d';
import {Size} from './Size.d';
import {TermCriteria} from './TermCriteria.d';
import {Vec2} from './Vec2.d';
import {Vec3} from './Vec3.d';
import {Vec4} from './Vec4.d';

export class Mat {
  readonly rows: number;
  readonly cols: number;
  readonly type: number;
  readonly channels: number;
  readonly depth: number;
  readonly dims: number;
  readonly empty: boolean;
  readonly step: number;
  readonly elemSize: number;
  readonly sizes: number[];
  constructor();
  constructor(channels: Mat[]);
  constructor(rows: number, cols: number, type: number);
  constructor(rows: number, cols: number, type: number, fillValue: number);
  constructor(rows: number, cols: number, type: number, fillValue: number[]);
  constructor(rows: number, cols: number, type: number, fillValue: number[]);
  constructor(rows: number, cols: number, type: number, fillValue: number[]);
  constructor(dataArray: number[][], type: number);
  constructor(dataArray: number[][][], type: number);
  constructor(dataArray: number[][][], type: number);
  constructor(dataArray: number[][][], type: number);
  constructor(data: Buffer, rows: number, cols: number, type?: number);
  abs(): Mat;
  absdiff(otherMat: Mat): Mat;
  adaptiveThreshold(
      maxVal: number, adaptiveMethod: number, thresholdType: number,
      blockSize: number, C: number): Mat;
  adaptiveThresholdAsync(
      maxVal: number, adaptiveMethod: number, thresholdType: number,
      blockSize: number, C: number): Promise<Mat>;
  add(otherMat: Mat): Mat;
  addWeighted(
      alpha: number, mat2: Mat, beta: number, gamma: number,
      dtype?: number): Mat;
  addWeightedAsync(
      alpha: number, mat2: Mat, beta: number, gamma: number,
      dtype?: number): Promise<Mat>;
  and(otherMat: Mat): Mat;
  at(row: number, col: number): number;
  at(row: number, col: number): Vec2;
  at(row: number, col: number): Vec3;
  at(row: number, col: number): Vec4;
  at(idx: number[]): number;
  at(idx: number[]): Vec2;
  at(idx: number[]): Vec3;
  at(idx: number[]): Vec4;
  atRaw(row: number, col: number): number;
  atRaw(row: number, col: number): number[];
  atRaw(row: number, col: number): number[];
  atRaw(row: number, col: number): number[];
  bgrToGray(): Mat;
  bgrToGrayAsync(): Promise<Mat>;
  bilateralFilter(
      d: number, sigmaColor: number, sigmaSpace: number,
      borderType?: number): Mat;
  bilateralFilterAsync(
      d: number, sigmaColor: number, sigmaSpace: number,
      borderType?: number): Promise<Mat>;
  bitwiseAnd(otherMat: Mat): Mat;
  bitwiseNot(): Mat;
  bitwiseOr(otherMat: Mat): Mat;
  bitwiseXor(otherMat: Mat): Mat;
  blur(kSize: Size, anchor?: Point2, borderType?: number): Mat;
  blurAsync(kSize: Size, anchor?: Point2, borderType?: number): Promise<Mat>;
  boxFilter(
      ddepth: number, ksize: Size, anchor?: Point2, normalize?: boolean,
      borderType?: number): Mat;
  boxFilterAsync(
      ddepth: number, ksize: Size, anchor?: Point2, normalize?: boolean,
      borderType?: number): Promise<Mat>;
  buildPyramid(maxLevel: number, borderType?: number): Mat[];
  buildPyramidAsync(maxLevel: number, borderType?: number): Promise<Mat[]>;
  calibrationMatrixValues(
      imageSize: Size, apertureWidth: number, apertureHeight: number): {
    fovx: number,
    fovy: number,
    focalLength: number,
    principalPoint: Point2,
    aspectRatio: number
  };
  calibrationMatrixValuesAsync(
      imageSize: Size, apertureWidth: number, apertureHeight: number): Promise<{
    fovx: number,
    fovy: number,
    focalLength: number,
    principalPoint: Point2,
    aspectRatio: number
  }>;
  canny(
      threshold1: number, threshold2: number, apertureSize?: number,
      L2gradient?: boolean): Mat;
  cannyAsync(
      threshold1: number, threshold2: number, apertureSize?: number,
      L2gradient?: boolean): Promise<Mat>;
  compareHist(H2: Mat, method: number): number;
  compareHistAsync(H2: Mat, method: number): Promise<number>;
  connectedComponents(connectivity?: number, ltype?: number): Mat;
  connectedComponentsAsync(connectivity?: number, ltype?: number): Promise<Mat>;
  connectedComponentsWithStats(connectivity?: number, ltype?: number):
      {labels: Mat, stats: Mat, centroids: Mat};
  connectedComponentsWithStatsAsync(connectivity?: number, ltype?: number):
      Promise<{labels: Mat, stats: Mat, centroids: Mat}>;
  convertScaleAbs(alpha: number, beta: number): Mat;
  convertScaleAbsAsync(alpha: number, beta: number): Promise<Mat>;
  convertTo(type: number, alpha?: number, beta?: number): Mat;
  convertToAsync(type: number, alpha?: number, beta?: number): Promise<Mat>;
  copy(mask?: Mat): Mat;
  copyAsync(mask?: Mat): Promise<Mat>;
  copyMakeBorder(
      top: number, bottom: number, left: number, right: number,
      borderType?: number, value?: number|Vec2|Vec3|Vec4): Mat;
  copyMakeBorderAsync(
      top: number, bottom: number, left: number, right: number,
      borderType?: number, value?: number|Vec2|Vec3|Vec4): Promise<Mat>;
  copyTo(dst: Mat, mask?: Mat): Mat;
  copyToAsync(dst: Mat, mask?: Mat): Promise<Mat>;
  cornerEigenValsAndVecs(
      blockSize: number, ksize?: number, borderType?: number): Mat;
  cornerEigenValsAndVecsAsync(
      blockSize: number, ksize?: number, borderType?: number): Promise<Mat>;
  cornerHarris(
      blockSize: number, ksize: number, k: number, borderType?: number): Mat;
  cornerHarrisAsync(
      blockSize: number, ksize: number, k: number,
      borderType?: number): Promise<Mat>;
  cornerMinEigenVal(blockSize: number, ksize?: number, borderType?: number):
      Mat;
  cornerMinEigenValAsync(
      blockSize: number, ksize?: number, borderType?: number): Promise<Mat>;
  cornerSubPix(
      corners: Point2[], winSize: Size, zeroZone: Size,
      criteria: TermCriteria): Point2[];
  cornerSubPixAsync(
      corners: Point2[], winSize: Size, zeroZone: Size,
      criteria: TermCriteria): Promise<Point2[]>;
  correctMatches(points1: Point2[], points2: Point2[]):
      {newPoints1: Point2[], newPoints2: Point2[]};
  correctMatchesAsync(points1: Point2[], points2: Point2[]):
      Promise<{newPoints1: Point2[], newPoints2: Point2[]}>;
  countNonZero(): number;
  countNonZeroAsync(): Promise<number>;
  cvtColor(code: number, dstCn?: number): Mat;
  cvtColorAsync(code: number, dstCn?: number): Promise<Mat>;
  dct(flags?: number): Mat;
  dctAsync(flags?: number): Promise<Mat>;
  decomposeEssentialMat(): {R1: Mat, R2: Mat, T: Vec3};
  decomposeEssentialMatAsync(): Promise<{R1: Mat, R2: Mat, T: Vec3}>;
  decomposeHomographyMat(K: Mat): {
    returnValue: number,
    rotations: Mat[],
    translations: Mat[],
    normals: Mat[]
  };
  decomposeHomographyMatAsync(K: Mat): Promise<{
    returnValue: number,
    rotations: Mat[],
    translations: Mat[],
    normals: Mat[]
  }>;
  decomposeProjectionMatrix(): {
    cameraMatrix: Mat,
    rotMatrix: Mat,
    transVect: Vec4,
    rotMatrixX: Mat,
    rotMatrixY: Mat,
    rotMatrixZ: Mat,
    eulerAngles: Mat
  };
  decomposeProjectionMatrixAsync(): Promise<{
    cameraMatrix: Mat,
    rotMatrix: Mat,
    transVect: Vec4,
    rotMatrixX: Mat,
    rotMatrixY: Mat,
    rotMatrixZ: Mat,
    eulerAngles: Mat
  }>;
  determinant(): number;
  dft(flags?: number, nonzeroRows?: number): Mat;
  dftAsync(flags?: number, nonzeroRows?: number): Promise<Mat>;
  dilate(
      kernel: Mat, anchor?: Point2, iterations?: number,
      borderType?: number): Mat;
  dilateAsync(
      kernel: Mat, anchor?: Point2, iterations?: number,
      borderType?: number): Promise<Mat>;
  distanceTransform(distanceType: number, maskSize: number, dstType?: number):
      Mat;
  distanceTransformAsync(
      distanceType: number, maskSize: number, dstType?: number): Promise<Mat>;
  distanceTransformWithLabels(
      distanceType: number, maskSize: number,
      labelType?: number): {labels: Mat, dist: Mat};
  distanceTransformWithLabelsAsync(
      distanceType: number, maskSize: number,
      labelType?: number): Promise<{labels: Mat, dist: Mat}>;
  div(s: number): Mat;
  dot(): Mat;
  drawArrowedLine(
      pt0: Point2, pt1: Point2, color?: Vec3, thickness?: number,
      lineType?: number, shift?: number, tipLength?: number): void;
  drawChessboardCorners(
      patternSize: Size, corners: Point2[], patternWasFound: boolean): void;
  drawChessboardCornersAsync(
      patternSize: Size, corners: Point2[],
      patternWasFound: boolean): Promise<void>;
  drawCircle(
      center: Point2, radius: number, color?: Vec3, thickness?: number,
      lineType?: number, shift?: number): void;
  drawContours(
      contours: Contour[], color: Vec3, contourIdx?: number, maxLevel?: number,
      offset?: Point2, lineType?: number, thickness?: number,
      shift?: number): void;
  drawEllipse(
      box: RotatedRect, color?: Vec3, thickness?: number,
      lineType?: number): void;
  drawEllipse(
      center: Point2, axes: Size, angle: number, startAngle: number,
      endAngle: number, color?: Vec3, thickness?: number, lineType?: number,
      shift?: number): void;
  drawFillConvexPoly(
      pts: Point2[], color?: Vec3, lineType?: number, shift?: number): void;
  drawFillPoly(
      pts: Point2[][], color?: Vec3, lineType?: number, shift?: number,
      offset?: Point2): void;
  drawLine(
      pt0: Point2, pt1: Point2, color?: Vec3, thickness?: number,
      lineType?: number, shift?: number): void;
  drawPolylines(
      pts: Point2[][], isClosed: boolean, color?: Vec3, thickness?: number,
      lineType?: number, shift?: number): void;
  drawRectangle(
      pt0: Point2, pt1: Point2, color?: Vec3, thickness?: number,
      lineType?: number, shift?: number): void;
  drawRectangle(
      rect: Rect, color?: Vec3, thickness?: number, lineType?: number,
      shift?: number): void;
  eigen(): Mat;
  eigenAsync(): Promise<Mat>;
  equalizeHist(): Mat;
  equalizeHistAsync(): Promise<Mat>;
  erode(kernel: Mat, anchor?: Point2, iterations?: number, borderType?: number):
      Mat;
  erodeAsync(
      kernel: Mat, anchor?: Point2, iterations?: number,
      borderType?: number): Promise<Mat>;
  exp(): Mat;
  log(): Mat;
  filter2D(
      ddepth: number, kernel: Mat, anchor?: Point2, delta?: number,
      borderType?: number): Mat;
  filter2DAsync(
      ddepth: number, kernel: Mat, anchor?: Point2, delta?: number,
      borderType?: number): Promise<Mat>;
  filterSpeckles(newVal: number, maxSpeckleSize: number, maxDiff: number):
      {newPoints1: Point2[], newPoints2: Point2[]};
  filterSpecklesAsync(newVal: number, maxSpeckleSize: number, maxDiff: number):
      Promise<{newPoints1: Point2[], newPoints2: Point2[]}>;
  find4QuadCornerSubpix(corners: Point2[], regionSize: Size): boolean;
  find4QuadCornerSubpixAsync(corners: Point2[], regionSize: Size):
      Promise<boolean>;
  findChessboardCorners(patternSize: Size, flags?: number):
      {returnValue: boolean, corners: Point2[]};
  findChessboardCornersAsync(patternSize: Size, flags?: number):
      Promise<{returnValue: boolean, corners: Point2[]}>;
  findContours(mode: number, method: number, offset?: Point2): Contour[];
  findContoursAsync(mode: number, method: number, offset?: Point2):
      Promise<Contour[]>;
  findEssentialMat(
      points1: Point2[], points2: Point2[], method?: number, prob?: number,
      threshold?: number): {E: Mat, mask: Mat};
  findEssentialMatAsync(
      points1: Point2[], points2: Point2[], method?: number, prob?: number,
      threshold?: number): Promise<{E: Mat, mask: Mat}>;
  findNonZero(): Point2[];
  findNonZeroAsync(): Promise<Point2[]>;
  flattenFloat(rows: number, cols: number): Mat;
  flip(flipCode: number): Mat;
  flipAsync(flipCode: number): Promise<Mat>;
  floodFill(
      seedPoint: Point2, newVal: number, mask?: Mat, loDiff?: number,
      upDiff?: number, flags?: number): {returnValue: number, rect: Rect};
  floodFill(
      seedPoint: Point2, newVal: Vec3, mask?: Mat, loDiff?: Vec3, upDiff?: Vec3,
      flags?: number): {returnValue: number, rect: Rect};
  floodFillAsync(
      seedPoint: Point2, newVal: number, mask?: Mat, loDiff?: number,
      upDiff?: number,
      flags?: number): Promise<{returnValue: number, rect: Rect}>;
  floodFillAsync(
      seedPoint: Point2, newVal: Vec3, mask?: Mat, loDiff?: Vec3, upDiff?: Vec3,
      flags?: number): Promise<{returnValue: number, rect: Rect}>;
  gaussianBlur(
      kSize: Size, sigmaX: number, sigmaY?: number, borderType?: number): Mat;
  gaussianBlurAsync(
      kSize: Size, sigmaX: number, sigmaY?: number,
      borderType?: number): Promise<Mat>;
  getData(): Buffer;
  getDataAsync(): Promise<Buffer>;
  getDataAsArray(): number[][];
  getDataAsArray(): number[][][];
  getDataAsArray(): number[][][];
  getDataAsArray(): number[][][];
  getOptimalNewCameraMatrix(
      distCoeffs: number[], imageSize: Size, alpha: number, newImageSize?: Size,
      centerPrincipalPoint?: boolean): {out: Mat, validPixROI: Rect};
  getOptimalNewCameraMatrixAsync(
      distCoeffs: number[], imageSize: Size, alpha: number, newImageSize?: Size,
      centerPrincipalPoint?: boolean): Promise<{out: Mat, validPixROI: Rect}>;
  getRegion(region: Rect): Mat;
  goodFeaturesToTrack(
      maxCorners: number, qualityLevel: number, minDistance: number, mask?: Mat,
      blockSize?: number, gradientSize?: number, useHarrisDetector?: boolean,
      harrisK?: number): Point2[];
  goodFeaturesToTrackAsync(
      maxCorners: number, qualityLevel: number, minDistance: number, mask?: Mat,
      blockSize?: number, gradientSize?: number, useHarrisDetector?: boolean,
      harrisK?: number): Promise<Point2[]>;
  grabCut(
      mask: Mat, rect: Rect, bgdModel: Mat, fgdModel: Mat, iterCount: number,
      mode: number): void;
  grabCutAsync(
      mask: Mat, rect: Rect, bgdModel: Mat, fgdModel: Mat, iterCount: number,
      mode: number): Promise<void>;
  guidedFilter(guide: Mat, radius: number, eps: number, ddepth?: number): Mat;
  guidedFilterAsync(guide: Mat, radius: number, eps: number, ddepth?: number):
      Promise<Mat>;
  hDiv(otherMat: Mat): Mat;
  hMul(otherMat: Mat): Mat;
  houghCircles(
      method: number, dp: number, minDist: number, param1?: number,
      param2?: number, minRadius?: number, maxRadius?: number): Vec3[];
  houghCirclesAsync(
      method: number, dp: number, minDist: number, param1?: number,
      param2?: number, minRadius?: number, maxRadius?: number): Promise<Vec3[]>;
  houghLines(
      rho: number, theta: number, threshold: number, srn?: number, stn?: number,
      min_theta?: number, max_theta?: number): Vec2[];
  houghLinesAsync(
      rho: number, theta: number, threshold: number, srn?: number, stn?: number,
      min_theta?: number, max_theta?: number): Promise<Vec2[]>;
  houghLinesP(
      rho: number, theta: number, threshold: number, minLineLength?: number,
      maxLineGap?: number): Vec4[];
  houghLinesPAsync(
      rho: number, theta: number, threshold: number, minLineLength?: number,
      maxLineGap?: number): Promise<Vec4[]>;
  idct(flags?: number): Mat;
  idctAsync(flags?: number): Promise<Mat>;
  idft(flags?: number, nonzeroRows?: number): Mat;
  idftAsync(flags?: number, nonzeroRows?: number): Promise<Mat>;
  inRange(lower: number, upper: number): Mat;
  inRange(lower: Vec3, upper: Vec3): Mat;
  inRangeAsync(lower: number, upper: number): Promise<Mat>;
  inRangeAsync(lower: Vec3, upper: Vec3): Promise<Mat>;
  integral(sdepth?: number, sqdepth?: number):
      {sum: Mat, sqsum: Mat, tilted: Mat};
  integralAsync(sdepth?: number, sqdepth?: number):
      Promise<{sum: Mat, sqsum: Mat, tilted: Mat}>;
  laplacian(
      ddepth: number, ksize?: number, scale?: number, delta?: number,
      borderType?: number): Mat;
  laplacianAsync(
      ddepth: number, ksize?: number, scale?: number, delta?: number,
      borderType?: number): Promise<Mat>;
  matMul(B: Mat): Mat;
  matMulDeriv(B: Mat): {dABdA: Mat, dABdB: Mat};
  matMulDerivAsync(B: Mat): Promise<{dABdA: Mat, dABdB: Mat}>;
  matchTemplate(template: Mat, method: number, mask?: Mat): Mat;
  matchTemplateAsync(template: Mat, method: number, mask?: Mat): Promise<Mat>;
  mean(): Vec4;
  meanAsync(): Promise<Vec4>;
  meanStdDev(mask?: Mat): {mean: Mat, stddev: Mat};
  meanStdDevAsync(mask?: Mat): Promise<{mean: Mat, stddev: Mat}>;
  medianBlur(kSize: number): Mat;
  medianBlurAsync(kSize: number): Promise<Mat>;
  minMaxLoc(mask?: Mat):
      {minVal: number, maxVal: number, minLoc: Point2, maxLoc: Point2};
  minMaxLocAsync(mask?: Mat):
      Promise<{minVal: number, maxVal: number, minLoc: Point2, maxLoc: Point2}>;
  moments(): Moments;
  momentsAsync(): Promise<Moments>;
  morphologyEx(
      kernel: Mat, morphType: number, anchor?: Point2, iterations?: number,
      borderType?: number): Mat;
  morphologyExAsync(
      kernel: Mat, morphType: number, anchor?: Point2, iterations?: number,
      borderType?: number): Promise<Mat>;
  mul(s: number): Mat;
  mulSpectrums(mat2: Mat, dftRows?: boolean, conjB?: boolean): Mat;
  mulSpectrumsAsync(mat2: Mat, dftRows?: boolean, conjB?: boolean):
      Promise<Mat>;
  norm(src2: Mat, normType?: number, mask?: Mat): number;
  norm(normType?: number, mask?: Mat): number;
  normalize(
      alpha?: number, beta?: number, normType?: number, dtype?: number,
      mask?: Mat): Mat;
  or(otherMat: Mat): Mat;
  padToSquare(color: Vec3): Mat;
  perspectiveTransform(m: Mat): Mat;
  perspectiveTransformAsync(m: Mat): Promise<Mat>;
  pop_back(numRows?: number): Mat;
  pop_backAsync(numRows?: number): Promise<Mat>;
  popBack(numRows?: number): Mat;
  popBackAsync(numRows?: number): Promise<Mat>;
  push_back(mat: Mat): Mat;
  push_backAsync(mat: Mat): Promise<Mat>;
  pushBack(mat: Mat): Mat;
  pushBackAsync(mat: Mat): Promise<Mat>;
  putText(
      text: string, origin: Point2, fontFace: number, fontScale: number,
      color?: Vec3, lineType?: number, thickness?: number,
      bottomLeftOrigin?: boolean): void;
  pyrDown(size?: Size, borderType?: number): Mat;
  pyrDownAsync(size?: Size, borderType?: number): Promise<Mat>;
  pyrUp(size?: Size, borderType?: number): Mat;
  pyrUpAsync(size?: Size, borderType?: number): Promise<Mat>;
  recoverPose(E: Mat, points1: Point2[], points2: Point2[], mask?: Mat):
      {returnValue: number, R: Mat, T: Vec3};
  recoverPoseAsync(E: Mat, points1: Point2[], points2: Point2[], mask?: Mat):
      Promise<{returnValue: number, R: Mat, T: Vec3}>;
  rectify3Collinear(
      distCoeffs1: number[], cameraMatrix2: Mat, distCoeffs2: number[],
      cameraMatrix3: Mat, distCoeffs3: number[], imageSize: Size, R12: Mat,
      T12: Vec3, R13: Mat, T13: Vec3, alpha: number, newImageSize: Size,
      flags: number): {
    returnValue: number,
    R1: Mat,
    R2: Mat,
    R3: Mat,
    P1: Mat,
    P2: Mat,
    P3: Mat,
    Q: Mat,
    roi1: Rect,
    roi2: Rect
  };
  rectify3CollinearAsync(
      distCoeffs1: number[], cameraMatrix2: Mat, distCoeffs2: number[],
      cameraMatrix3: Mat, distCoeffs3: number[], imageSize: Size, R12: Mat,
      T12: Vec3, R13: Mat, T13: Vec3, alpha: number, newImageSize: Size,
      flags: number): Promise<{
    returnValue: number,
    R1: Mat,
    R2: Mat,
    R3: Mat,
    P1: Mat,
    P2: Mat,
    P3: Mat,
    Q: Mat,
    roi1: Rect,
    roi2: Rect
  }>;
  reduce(dim: number, rtype: number, dtype?: number): Mat;
  reduceAsync(dim: number, rtype: number, dtype?: number): Promise<Mat>;
  reprojectImageTo3D(Q: Mat, handleMissingValues?: boolean, ddepth?: number):
      Mat;
  reprojectImageTo3DAsync(
      Q: Mat, handleMissingValues?: boolean, ddepth?: number): Promise<Mat>;
  rescale(factor: number): Mat;
  rescaleAsync(factor: number): Promise<Mat>;
  resize(
      rows: number, cols: number, fx?: number, fy?: number,
      interpolation?: number): Mat;
  resize(dsize: Size, fx?: number, fy?: number, interpolation?: number): Mat;
  resizeAsync(
      rows: number, cols: number, fx?: number, fy?: number,
      interpolation?: number): Promise<Mat>;
  resizeAsync(dsize: Size, fx?: number, fy?: number, interpolation?: number):
      Promise<Mat>;
  resizeToMax(maxRowsOrCols: number): Mat;
  resizeToMaxAsync(maxRowsOrCols: number): Promise<Mat>;
  rodrigues(): {dst: Mat, jacobian: Mat};
  rodriguesAsync(): Promise<{dst: Mat, jacobian: Mat}>;
  rotate(rotateCode: number): Mat;
  rotateAsync(rotateCode: number): Promise<Mat>;
  rqDecomp3x3():
      {returnValue: Vec3, mtxR: Mat, mtxQ: Mat, Qx: Mat, Qy: Mat, Qz: Mat};
  rqDecomp3x3Async(): Promise<
      {returnValue: Vec3, mtxR: Mat, mtxQ: Mat, Qx: Mat, Qy: Mat, Qz: Mat}>;
  scharr(
      ddepth: number, dx: number, dy: number, scale?: number, delta?: number,
      borderType?: number): Mat;
  scharrAsync(
      ddepth: number, dx: number, dy: number, scale?: number, delta?: number,
      borderType?: number): Promise<Mat>;
  sepFilter2D(
      ddepth: number, kernelX: Mat, kernelY: Mat, anchor?: Point2,
      delta?: number, borderType?: number): Mat;
  sepFilter2DAsync(
      ddepth: number, kernelX: Mat, kernelY: Mat, anchor?: Point2,
      delta?: number, borderType?: number): Promise<Mat>;
  set(row: number, col: number, value: number): void;
  set(row: number, col: number, value: number[]): void;
  set(row: number, col: number, value: number[]): void;
  set(row: number, col: number, value: number[]): void;
  set(row: number, col: number, value: Vec2): void;
  set(row: number, col: number, value: Vec3): void;
  set(row: number, col: number, value: Vec4): void;
  setTo(value: number, mask?: Mat): Mat;
  setTo(value: Vec2, mask?: Mat): Mat;
  setTo(value: Vec3, mask?: Mat): Mat;
  setTo(value: Vec4, mask?: Mat): Mat;
  setToAsync(value: number, mask?: Mat): Promise<Mat>;
  setToAsync(value: Vec2, mask?: Mat): Promise<Mat>;
  setToAsync(value: Vec3, mask?: Mat): Promise<Mat>;
  setToAsync(value: Vec4, mask?: Mat): Promise<Mat>;
  sobel(
      ddepth: number, dx: number, dy: number, ksize?: number, scale?: number,
      delta?: number, borderType?: number): Mat;
  sobelAsync(
      ddepth: number, dx: number, dy: number, ksize?: number, scale?: number,
      delta?: number, borderType?: number): Promise<Mat>;
  solve(mat2: Mat, flags?: number): Mat;
  solveAsync(mat2: Mat, flags?: number): Promise<Mat>;
  split(): Mat[];
  splitAsync(): Promise<Mat[]>;
  splitChannels(): Mat[];
  splitChannelsAsync(): Promise<Mat[]>;
  sqrBoxFilter(
      ddepth: number, ksize: Size, anchor?: Point2, normalize?: boolean,
      borderType?: number): Mat;
  sqrBoxFilterAsync(
      ddepth: number, ksize: Size, anchor?: Point2, normalize?: boolean,
      borderType?: number): Promise<Mat>;
  sqrt(): Mat;
  stereoRectify(
      distCoeffs1: number[], cameraMatrix2: Mat, distCoeffs2: number[],
      imageSize: Size, R: Mat, T: Vec3, flags?: number, alpha?: number,
      newImageSize?: Size):
      {R1: Mat, R2: Mat, P1: Mat, P2: Mat, Q: Mat, roi1: Rect, roi2: Rect};
  stereoRectifyAsync(
      distCoeffs1: number[], cameraMatrix2: Mat, distCoeffs2: number[],
      imageSize: Size, R: Mat, T: Vec3, flags?: number, alpha?: number,
      newImageSize?: Size):
      Promise<
          {R1: Mat, R2: Mat, P1: Mat, P2: Mat, Q: Mat, roi1: Rect, roi2: Rect}>;
  sub(otherMat: Mat): Mat;
  sum(): number;
  sum(): Vec2;
  sum(): Vec3;
  sum(): Vec4;
  sumAsync(): Promise<number>;
  sumAsync(): Promise<Vec2>;
  sumAsync(): Promise<Vec3>;
  sumAsync(): Promise<Vec4>;
  threshold(thresh: number, maxVal: number, type: number): Mat;
  thresholdAsync(thresh: number, maxVal: number, type: number): Promise<Mat>;
  transform(m: Mat): Mat;
  transformAsync(m: Mat): Promise<Mat>;
  transpose(): Mat;
  triangulatePoints(projPoints1: Point2[], projPoints2: Point2[]): Mat;
  triangulatePointsAsync(projPoints1: Point2[], projPoints2: Point2[]):
      Promise<Mat>;
  undistort(cameraMatrix: Mat, distCoeffs: Mat): Mat;
  undistortAsync(cameraMatrix: Mat, distCoeffs: Mat): Promise<Mat>;
  validateDisparity(
      cost: Mat, minDisparity: number, numberOfDisparities: number,
      disp12MaxDisp?: number): void;
  validateDisparityAsync(
      cost: Mat, minDisparity: number, numberOfDisparities: number,
      disp12MaxDisp?: number): Promise<void>;
  warpAffine(
      transforMationMatrix: Mat, size?: Size, flags?: number,
      borderMode?: number, borderValue?: Vec3): Mat;
  warpAffineAsync(
      transforMationMatrix: Mat, size?: Size, flags?: number,
      borderMode?: number, borderValue?: Vec3): Promise<Mat>;
  warpPerspective(
      transforMationMatrix: Mat, size?: Size, flags?: number,
      borderMode?: number, borderValue?: Vec3): Mat;
  warpPerspectiveAsync(
      transforMationMatrix: Mat, size?: Size, flags?: number,
      borderMode?: number, borderValue?: Vec3): Promise<Mat>;
  watershed(markers: Mat): Mat;
  watershedAsync(markers: Mat): Promise<Mat>;
  release(): void;

  static eye(rows: number, cols: number, type: number): Mat;
}
