import * as tf from '@tensorflow/tfjs-core';
import * as bodyPix from './src/index';
import * as fs from 'fs';

// tslint:disable-next-line:no-require-imports
const {testImage} = require('./testdata/test_image0.js');

// console.log(testImage);

// const dataIn = new Uint8Array(testImage.data);

// const t = tf.tensor(dataIn, [testImage.height, testImage.width, 3]);
// t.print();

// fs.writeFileSync('./testdata/test_image0.bin', dataIn);

// const back = fs.readFileSync('./testdata/test_image0.bin');
// const farray = new Uint8Array(back.buffer);
// console.log('reconstructed', farray);
// console.log('both', farray[0], dataIn[0]);

// console.log(t.shape);

async function go() {
    const img = tf.tensor3d(testImage.data, [testImage.height, testImage.width, 3]);
    const net = await bodyPix.load();
    const partSegmentation = await net.segmentMultiPersonParts(img);

    console.log(partSegmentation);
}

go();