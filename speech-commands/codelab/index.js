'use strict';

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

let model;
let recognizer;
let examples = [];

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

function collect(label) {
  if (label == null) {
    return recognizer.stopListening();
  }
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    examples.push({vals, label});
    document.querySelector('#console').textContent =
        `${examples.length} examples collected`;
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}

function toggleButtons(enable) {
  document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

async function train() {
  toggleButtons(false);
  const ys = tf.oneHot(examples.map(e => e.label), 3);
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector('#console').textContent =
            `Accuracy: ${(logs.acc * 100).toFixed(1)}%`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}

async function moveSlider(labelTensor) {
  const label = (await labelTensor.data())[0];
  if (label == 2) {
    return;
  }
  let delta = 0.1;
  const prevValue = +document.getElementById('output').value;
  document.getElementById('output').value =
      prevValue + delta * (label === 0 ? 1 : -1);
}

function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById('listen').textContent = 'Listen';
    return;
  }
  toggleButtons(false);
  document.getElementById('listen').textContent = 'Stop';
  document.getElementById('listen').disabled = false;

  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    const probs = model.predict(input);
    const predLabel = probs.argMax(1);
    await moveSlider(predLabel);
    tf.dispose([input, probs, predLabel]);
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}

function buildModel() {
  model = tf.sequential();
  model.add(tf.layers.depthwiseConv2d({
    depthMultiplier: 8,
    kernelSize: [NUM_FRAMES, 3],
    activation: 'relu',
    inputShape: INPUT_SHAPE
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
}

async function app() {
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  buildModel();
}

// function predictWord() {
//   // Array of words that the recognizer is trained to recognize.
//   const words = recognizer.wordLabels();
//   console.log(words);
//   recognizer.listen(({scores}) => {
//     // Turn scores into a list of (score,word) pairs.
//     scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
//     // Find the most probable word.
//     scores.sort((s1, s2) => s2.score - s1.score);
//     document.querySelector('#console').textContent = scores[0].word;
//   }, {
//     probabilityThreshold: 0.75
//   });
// }

// async function app() {
//   recognizer = speechCommands.create('BROWSER_FFT');
//   await recognizer.ensureModelLoaded();
//   predictWord();
// }

app();
