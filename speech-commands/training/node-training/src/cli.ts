#!/usr/bin/env node
// Load the binding
import '@tensorflow/tfjs-node-gpu';

import chalk from 'chalk';
import * as ora from 'ora';
import {AudioModel} from './audio_model';

import * as Vorpal from 'vorpal';

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
vorpal.command('create [labels...]')
    .alias('c')
    .description('create the audio model')
    .action((args, cb) => {
      console.log(args.labels);
      labels = args.labels as string[];
      model = new AudioModel(MODEL_SHAPE, labels);
      cb();
    });

vorpal
    .command(
        'load all <dir>',
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
        'load <dir> <label>',
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
vorpal.command('save <filename>')
    .alias('s')
    .description('save the audio model')
    .action((args) => {
      spinner.start(`saving to ${args.filename} ...`);
      return model.save(args.filename as string).then(() => {
        spinner.succeed(`${args.filename} saved.`);
      }, () => spinner.fail(`failed to save ${args.filename}`));
    });

vorpal.show();
