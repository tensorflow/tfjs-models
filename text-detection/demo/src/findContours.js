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
