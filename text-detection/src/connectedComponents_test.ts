// import * as tf from '@tensorflow/tfjs';
// import {readFileSync} from 'fs';
// import {decode} from 'jpeg-js';
// import {resolve} from 'path';

// import {Point} from './geometry';
// import {minAreaRect} from './minAreaRect';
// import cv from './opencv';

describe('connectedComponents', () => {
  it('The connectedComponents output coincides with OpenCV results.',
     () => {
         // const input = tf.tidy(() => {
         //   const testImage =
         //       decode(readFileSync(resolve(__dirname,
         //       'assets/example.jpeg')), true);
         //   const rawData = tf.tensor(testImage.data, [
         //                       testImage.height, testImage.width, 4
         //                     ]).arraySync() as number[][][];
         //   const inputBuffer =
         //       tf.buffer([testImage.height, testImage.width], 'int32');
         //   for (let columnIndex = 0; columnIndex < testImage.height;
         //   ++columnIndex) {
         //     for (let rowIndex = 0; rowIndex < testImage.width; ++rowIndex) {
         //       for (let channel = 0; channel < 3; ++channel) {
         //         inputBuffer.set(
         //             rawData[columnIndex][rowIndex][channel], columnIndex,
         //             rowIndex);
         //       }
         //     }
         //   }

         //   return inputBuffer.toTensor();
         // }) as tf.Tensor2D;
         // const inputArray = input.arraySync();
         // const inputData = input.dataSync();
     });
});
