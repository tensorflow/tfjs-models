/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as tf from '@tensorflow/tfjs';
import * as argparse from 'argparse';
import * as fs from 'fs';
import * as path from 'path';
import * as shelljs from 'shelljs';

import {Dataset} from '../../src/dataset';

/**
 * Create TensorFlow.js Model for speech-commands recognition.
 *
 * @param inputShape Shape of the input, expected to be a rank-3 shape:
 *   [numFrames, frameSize, 1].
 *   numFrames is the number of frames in each spectrogram input.
 *   frameSize is the size of the frequency dimension of the spectrogram.
 *   1 is the dummy "channels" dimension.
 * @param numClasses Number of output classes.
 * @returns An uncompiled tf.Model object.
 */
export function createBrowserFFTModel(
    inputShape: tf.Shape, numClasses: number): tf.Model {
  const model = tf.sequential();
  model.add(tf.layers.conv2d(
      {filters: 8, kernelSize: [2, 8], activation: 'relu', inputShape}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(
      tf.layers.conv2d({filters: 32, kernelSize: [2, 4], activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(
      tf.layers.conv2d({filters: 32, kernelSize: [2, 4], activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(
      tf.layers.conv2d({filters: 32, kernelSize: [2, 4], activation: 'relu'}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [1, 2]}));
  model.add(tf.layers.flatten({}));
  model.add(tf.layers.dropout({rate: 0.25}));
  model.add(tf.layers.dense({units: 2000, activation: 'relu'}));
  model.add(tf.layers.dropout({rate: 0.5}));
  model.add(tf.layers.dense({units: numClasses, activation: 'softmax'}));

  return model;
}

/**
 * Load dataset from a processed data directory.
 *
 * @param inputPath Path to the input directory, expcted to contain
 *   a set of TFJSSCDS-format (.bin) files in a nested fashion. See
 *   `../../src/dataset.ts` for the definition of the TFJSSCDS format.
 * @return A Dataset object, which includes the data from all the TFJSSCDS
 *   (.bin) files under `inputPath` combined.
 */
function loadDataset(inputPath): Dataset {
  if (!fs.lstatSync(inputPath).isDirectory()) {
    throw new Error(`Input path is not a directory: ${inputPath}`);
  }

  const dataset = new Dataset();

  const dirContent = fs.readdirSync(inputPath);
  for (const item of dirContent) {
    const fullPath = path.join(inputPath, item);
    let incomingDataset: Dataset;
    if (fs.lstatSync(fullPath).isDirectory()) {
      // A directory. Call recursively.
      incomingDataset = loadDataset(fullPath);
    } else {
      // A file.
      const arrayBuffer = fs.readFileSync(fullPath).buffer;
      incomingDataset = new Dataset(arrayBuffer);
    }
    dataset.merge(incomingDataset);
  }

  return dataset;
}

function parseArguments(): any {
  const parser = new argparse.ArgumentParser(
      {description: 'Training a browser-FFT speech-commands model'});
  parser.addArgument('dataPath', {
    type: 'string',
    help: 'Path to an input data directory. The directory is expected to ' +
        'contain .bin files in the TFJSSCDS format in a nested fashion.'
  });
  parser.addArgument('modelSavePath', {
    type: 'string',
    help: 'Path to directory in which the trained model will be saved, ' +
        'e.g., "./my_18w_model'
  });
  parser.addArgument('--gpu', {
    action: 'storeTrue',
    help: 'Perform training using tfjs-node-gpu (Requires CUDA and CuDNN)'
  });
  parser.addArgument(
      '--epochs',
      {type: 'int', defaultValue: 200, help: 'Number of training epochs'});
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 512, help: 'Batch size for training'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.1,
    help: 'Validation split for training'
  });
  parser.addArgument('--optimizer', {
    type: 'string',
    defaultValue: 'rmsprop',
    help: 'Optimizer name for training (e.g., adam, rmsprop)'
  });
  parser.addArgument(
      '--learningRate',
      {type: 'float', defaultValue: 1e-3, help: 'Learning rate for training'});
  return parser.parseArgs();
}

async function main() {
  const args = parseArguments();

  if (args.gpu) {
    console.log('Training using GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    require('@tensorflow/tfjs-node');
  }

  const trainDataDir = path.join(args.dataPath, 'train');
  console.log(`Loading trainig data from ${trainDataDir}...`);
  const trainDataset = loadDataset(trainDataDir);
  const vocab = trainDataset.getVocabulary();
  tf.util.assert(
      vocab.length > 1,
      `Expected vocabulary to have at least two words, but ` +
          `got vocabulary: ${JSON.stringify(vocab)}`);
  console.log(`Vocabulary size: ${vocab.length} (${JSON.stringify(vocab)})`);

  console.log('Collecting spectrogram and targets data...');
  let {xs, ys} = trainDataset.getSpectrogramsAsTensors(null, {shuffle: true});
  tf.util.assert(
      xs.rank === 4,
      `Expected xs tensor to be rank-4, but got rank ${xs.rank}`);
  tf.util.assert(
      ys.rank === 2,
      `Expected ys tensor to be rank-2, but got rank ${ys.rank}`);

  // Split the data manually into the training and validation subsets.
  // We do this manually for memory efficiency.
  console.log('Splitting data into training and validation subsets...');
  let validationData: [tf.Tensor, tf.Tensor] = null;
  if (args.validationSplit > 0) {
    const numExamples = xs.shape[0];
    const xsData = xs.dataSync();
    const ysData = ys.dataSync();
    xs.dispose();
    ys.dispose();

    const numValExamples = Math.round(numExamples * args.validationSplit);
    const numTrainExamples = numExamples - numValExamples;
    console.log(`# of training examples: ${numTrainExamples}`);
    console.log(`# of validation examples: ${numValExamples}`);

    const spectrogramSize = xs.shape[1] * xs.shape[2];
    xs = tf.tensor4d(
        xsData.slice(0, numTrainExamples * spectrogramSize),
        [numTrainExamples, xs.shape[1], xs.shape[2], 1]);
    ys = tf.tensor2d(
        ysData.slice(0, numTrainExamples * ys.shape[1]),
        [numTrainExamples, ys.shape[1]]);
    const valXs = tf.tensor4d(
        xsData.slice(
            numTrainExamples * spectrogramSize, numExamples * spectrogramSize),
        [numValExamples, xs.shape[1], xs.shape[2], 1]);
    const valYs = tf.tensor2d(
        ysData.slice(numTrainExamples * ys.shape[1], numExamples * ys.shape[1]),
        [numValExamples, ys.shape[1]]);
    validationData = [valXs, valYs];
  }

  const model = createBrowserFFTModel(xs.shape.slice(1), vocab.length);
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train[args.optimizer](args.learningRate),
    metrics: ['accuracy']
  });
  model.summary();

  // Train the model.
  await model.fit(
      xs, ys, {epochs: args.epochs, batchSize: args.batchSize, validationData});
  tf.dispose([xs, ys, validationData]);  // For memory efficiency.

  // Evaluate the model.
  if (fs.existsSync(path.join(args.dataPath, 'test'))) {
    const testDataset = loadDataset(path.join(args.dataPath, 'test'));
    const {xs: testXs, ys: testYs} =
        testDataset.getSpectrogramsAsTensors(null, {shuffle: true});
    console.log(`\n# of test examples: ${testXs.shape[0]}`);

    const [testLossScalar, testAccScalar] =
        model.evaluate(testXs, testYs, {batchSize: args.batchSize}) as
        tf.Tensor[];
    const testLoss = (await testLossScalar.data())[0];
    const testAcc = (await testAccScalar.data())[0];
    console.log(`Test loss: ${testLoss.toFixed(6)}`);
    console.log(`Test accuracy: ${testAcc.toFixed(6)}`);
    tf.dispose([testXs, testYs]);
  }

  // Save the trained model.
  const saveDir = path.dirname(args.modelSavePath);
  if (!fs.existsSync(saveDir)) {
    shelljs.mkdir('-p', saveDir);
  }
  await model.save(`file://${args.modelSavePath}`);
  console.log(`Saved model to: ${args.modelSavePath}`);

  // Sava the metadata for the model.
  const metadata: {} = {words: vocab, frameSize: xs.shape[2]};
  const metadataPath = path.join(args.modelSavePath, 'metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata));
  console.log(`Saved metadata to: ${metadataPath}`);
}

if (require.main === module) {
  main();
}
