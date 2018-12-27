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
function createBrowserFFTModel(
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
 *   a set of TFJSSCDS-format (.bin) files in a nested fashion.
 * @return A Dataset objects, which includes the data from all the TFJSSCDS
 *   (.bin) files combined.
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
      console.log(`Recursive call @ ${fullPath}`);  // DEBUG
      incomingDataset = loadDataset(fullPath);
    } else {
      // A file.
      console.log(`Reading file ${fullPath}`);  // DEBUG
      const arrayBuffer = fs.readFileSync(fullPath).buffer;
      incomingDataset = new Dataset(arrayBuffer);
    }
    dataset.merge(incomingDataset);
  }

  return dataset;
}

async function main() {
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
      {type: 'int', defaultValue: 100, help: 'Number of training epochs'});
  parser.addArgument(
      '--batchSize',
      {type: 'int', defaultValue: 512, help: 'Batch size for training'});
  parser.addArgument('--validationSplit', {
    type: 'float',
    defaultValue: 0.15,
    help: 'Validation split for training'
  });
  parser.addArgument(
      '--learningRate',
      {type: 'float', defaultValue: 5e-4, help: 'Learning rate for training'});
  const args = parser.parseArgs();

  if (args.gpu) {
    console.log('Training using GPU.');
    require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Training using CPU.');
    require('@tensorflow/tfjs-node');
  }

  const dataset = loadDataset(args.dataPath);
  const vocab = dataset.getVocabulary();
  tf.util.assert(
      vocab.length > 1,
      `Expected vocabulary to have at least two words, but ` +
          `got vocabulary: ${JSON.stringify(vocab)}`);
  console.log(`vocab.length = ${vocab.length}`);  // DEBUG

  const {xs, ys} = dataset.getSpectrogramsAsTensors();
  tf.util.assert(
      xs.rank === 4,
      `Expected xs tensor to be rank-4, but got rank ${xs.rank}`);
  tf.util.assert(
      ys.rank === 2,
      `Expected ys tensor to be rank-2, but got rank ${ys.rank}`);
  console.log(xs.shape);  // DEBUG
  console.log(ys.shape);  // DEBUG

  ys.print(true);  // DEBUG

  const model = createBrowserFFTModel(xs.shape.slice(1), vocab.length);
  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: tf.train.adam(args.learningRate),
    metrics: ['accuracy']
  });
  model.summary();

  // Train the model.
  await model.fit(xs, ys, {
    epochs: args.epochs,
    batchSize: args.batchSize,
    validationSplit: args.validationSplit
  });

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

main();
