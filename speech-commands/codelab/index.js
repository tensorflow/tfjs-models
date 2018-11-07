'use strict';

let origModel;
let newModel;
let recognizer;
let spectrogram;
let isListening = false;
const activations = [];
const labels = [];

async function addExample(label) {
  const inputShape = recognizer.modelInputShape();
  inputShape[0] = 1;
  toggleButtons(false);
  const example = await spectrogram.collectExample('test');
  toggleButtons(true);
  const input = tf.tensor(example.data, inputShape);
  const activation = await origModel.predict(input);
  activations.push(activation.squeeze());
  labels.push(label);
}

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

async function train() {
  toggleButtons(false);
  await newModel.fit(tf.stack(activations), tf.oneHot(labels, 3), {
    batchSize: 1,
    epochs: 10,
    callbacks: {
      onBatchEnd: (batch, logs) => {
        console.log(batch, logs.loss.toFixed(3));
      }
    }
  });
  toggleButtons(true);
}

function showPrediction(label) {
  const classes = ['A', 'B', 'C'];
  document.getElementById('prediction').textContent = classes[label];
}

function listen() {
  if (isListening) {
    isListening = false;
    recognizer.stopStreaming();
    toggleButtons(true);
    document.getElementById('listen').textContent = 'Listen';
    return;
  }
  isListening = true;
  toggleButtons(false);
  document.getElementById('listen').textContent = 'Stop';
  document.getElementById('listen').disabled = false;
  // `startStreaming()` takes two arguments:
  // 1. A callback function that is invoked anytime a word is recognized.
  // 2. A configuration object with adjustable fields such a
  //    - includeSpectrogram
  //    - probabilityThreshold

  recognizer.startStreaming(async result => {
    const inputShape = recognizer.modelInputShape();
    inputShape[0] = 1;
    const input = tf.tensor(result.spectrogram.data, inputShape);
    const activation = await origModel.predict(input);
    const predictions = newModel.predict(activation);
    const probs = await predictions.data();
    const maxProb = probs.reduce((prev, curr) => {
      return Math.max(prev, curr);
    });
    if (maxProb < 0.8) {
      return showPrediction('');
    }
    const predictedTensor = predictions.argMax(1);
    const predictedLabel = (await predictedTensor.data())[0];
    showPrediction(predictedLabel);
  }, {
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true,
    overlapFactor: 0.9
  });

}

async function app() {
  console.log('Loading speech commands...')
  // Load the model.
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  console.log('Sucessfully loaded model');

  spectrogram = recognizer.createTransfer('codelab');
  origModel = tf.model({
    inputs: recognizer.model.inputs,
    outputs: recognizer.model.getLayer('dense_1').output
  });

  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));
  document.getElementById('train').addEventListener('click', () => train());
  document.getElementById('listen').addEventListener('click', () => listen());

  // Create a new model.
  newModel = tf.sequential();
  newModel.add(tf.layers.dense({
    units: 3,
    inputShape: [2000],
    activation: 'softmax'
  }));
  const optimizer = tf.train.sgd(0.001);
  newModel.compile({optimizer, loss: 'categoricalCrossentropy'});
}

app();
