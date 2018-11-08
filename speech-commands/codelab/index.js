'use strict';

let newModel;
let recognizer;
let isListening = false;
const activations = [];
const labels = [];

async function addExample(label) {
  toggleButtons(false);
  const example = await recognizer.recognize(null, {includeEmbedding: true});
  toggleButtons(true);
  activations.push(example.embedding);
  labels.push(label);
}

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

async function train() {
  toggleButtons(false);
  await newModel.fit(tf.concat(activations), tf.oneHot(labels, 3), {
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


  recognizer.startStreaming(async result => {
    const probs = newModel.predict(result.embedding);
    const maxProb = (await probs.max().data())[0];
    if (maxProb < 0.8) {
      return showPrediction('');
    }
    const predictedLabel = (await probs.argMax(1).data())[0];
    showPrediction(predictedLabel);
  }, {
    overlapFactor: 0.95,
    includeEmbedding: true
  });

}

async function app() {
  console.log('Loading speech commands...')
  // Load the model.
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  console.log('Sucessfully loaded model');

  // Setup the UI.
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
