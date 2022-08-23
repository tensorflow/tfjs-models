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

// Load the binding
import '@tensorflow/tfjs-node';

import chalk from 'chalk';
import * as ora from 'ora';
import * as Vorpal from 'vorpal';

import {AudioModel} from './audio_model';
import {Dataset} from './dataset';
import {WavFileFeatureExtractor} from './wav_file_feature_extractor';

// tslint:disable-next-line:no-any
(global as any).AudioContext = class AudioContext {};

export const MODEL_SHAPE = [98, 40, 1];
export const labelsMsg = [
  {type: 'input', name: 'labels', message: 'Enter labels (seperate by comma)'}
];
export const trainingMsg = [
  {type: 'input', name: 'dir', message: 'Enter file directory'},
  {type: 'input', name: 'label', message: 'Enter label for the directory'}
];
export const filenameMsg = [{
  type: 'input',
  name: 'filename',
  message: 'Enter target filename for the model'
}];
let model: AudioModel;
let labels: string[];
const vorpal = new Vorpal();
let spinner = ora();
vorpal.command('create_model [labels...]')
    .alias('c')
    .description('create the audio model')
    .action((args, cb) => {
      console.log(args.labels);
      labels = args.labels as string[];
      model = new AudioModel(
          MODEL_SHAPE, labels, new Dataset(labels.length),
          new WavFileFeatureExtractor());
      cb();
    });

vorpal
    .command(
        'load_dataset all <dir>',
        'Load all the data from the root directory by the labels')
    .alias('la')
    .action((args) => {
      spinner.start('load dataset ...');
      return model
          .loadAll(
              args.dir as string,
              (text: string, finished?: boolean) => {
                if (finished) {
                  spinner.succeed(text);
                } else {
                  spinner.start();
                  spinner.text = text;
                  spinner.render();
                }
              })
          .then(() => spinner.stop());
    });
vorpal
    .command(
        'load_dataset <dir> <label>',
        'Load the dataset from the directory with the label')
    .alias('l')
    .action((args) => {
      spinner = ora('creating tensors ...');
      spinner.start();
      return model
          .loadData(
              args.dir as string, args.label as string,
              (text: string) => {
                // console.log(text);
                spinner.text = text;
                spinner.render();
              })
          .then(() => spinner.stop(), (err) => {
            spinner.fail(`failed to load: ${err}`);
          });
    });
vorpal.command('dataset size', 'Show the size of the dataset')
    .alias('d')
    .action((args, cb) => {
      console.log(chalk.green(`dataset size = ${model.size()}`));
      cb();
    });
vorpal.command('train [epoch]')
    .alias('t')
    .description('train all audio dataset')
    .action((args) => {
      spinner = ora('training models ...').start();
      return model
          .train(parseInt(args.epoch as string, 10) || 20, {
            onBatchEnd: async (batch, logs) => {
              spinner.text = chalk.green(`loss: ${logs.loss.toFixed(5)}`);
              spinner.render();
            },
            onEpochEnd: async (epoch, logs) => {
              spinner.succeed(chalk.green(
                  `epoch: ${epoch}, loss: ${logs.loss.toFixed(5)}` +
                  `, accuracy: ${logs.acc.toFixed(5)}` +
                  `, validation accuracy: ${logs.val_acc.toFixed(5)}`));
              spinner.start();
            }
          })
          .then(() => spinner.stop());
    });
vorpal.command('save_model <filename>')
    .alias('s')
    .description('save the audio model')
    .action((args) => {
      spinner.start(`saving to ${args.filename} ...`);
      return model.save(args.filename as string).then(() => {
        spinner.succeed(`${args.filename} saved.`);
      }, () => spinner.fail(`failed to save ${args.filename}`));
    });

vorpal.show();
