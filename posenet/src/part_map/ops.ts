import * as tf from '@tensorflow/tfjs';

const toColorMap = (inputShape: number[]): tf.webgl.GPGPUProgram => ({
  variableNames: ['x', 'C'],
  outputShape: [...inputShape.slice(), 3],
  userCode: `
  void main() {
    ivec2 coords = getOutputCoords();

    int colorId = round(getX(coords.x));

    float colorValue = getC(colorId, coords.y);

    // if color id was -1, clear it as there was no color here.
    colorValue *= step(0., float(colorId));

    setOutput(colorValue);
  }
  `
});

export {toColorMap};
