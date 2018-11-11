'use strict';

const NUM_FRAMES = 3;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

let newModel;
let recognizer;
let examples = [];
let labels = [];

function save() {
  sessionStorage.setItem('examples',
      JSON.stringify(examples.map(vals => Array.from(vals))));
  sessionStorage.setItem('labels', JSON.stringify(labels));
  console.log(`Saved ${examples.length} examples.`);
}
function load() {
  examples = JSON.parse(sessionStorage.getItem('examples'))
    .map(vals => new Float32Array(vals));
  labels = JSON.parse(sessionStorage.getItem('labels'));
  console.log(`Loaded ${examples.length} examples.`);
}

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
    examples.push(vals);
    labels.push(label);
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
  const ys = tf.oneHot(labels, 3);
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples), xsShape);

  await newModel.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(epoch, logs.acc.toFixed(3));
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
    const probs = newModel.predict(input);
    const predLabel = probs.argMax(1);
    await moveSlider(predLabel);
    tf.dispose([input, probs, predLabel]);
  }, {
    overlapFactor: 0.999,
    includeSpectrogram: true,
    invokeCallbackOnNoiseAndUnknown: true
  });
}

async function app() {
  console.log('Loading speech commands...')
  // Load the model.
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  // Warmup.
  await recognizer.recognize(null, {includeEmbedding: true});
  console.log('Sucessfully loaded model');
  load();

  // Setup the UI.
  document.getElementById('up').onmousedown = () => collect(0);
  document.getElementById('up').onmouseup = () => collect(null);

  document.getElementById('down').onmousedown = () => collect(1);
  document.getElementById('down').onmouseup = () => collect(null);

  document.getElementById('noise').onmousedown = () => collect(2);
  document.getElementById('noise').onmouseup = () => collect(null);

  document.getElementById('train').onmousedown = () => train();
  document.getElementById('listen').onmouseup = () => listen();

  // Create a new model.
  newModel = tf.sequential();
  newModel.add(tf.layers.depthwiseConv2d(
    {depthMultiplier: 8, kernelSize: [NUM_FRAMES, 3], activation: 'relu', inputShape: INPUT_SHAPE}));
  newModel.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  newModel.add(tf.layers.depthwiseConv2d(
      {depthMultiplier: 2, kernelSize: [1, 3], activation: 'relu'}));
  newModel.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  newModel.add(tf.layers.depthwiseConv2d(
      {depthMultiplier: 2, kernelSize: [1, 3], activation: 'relu'}));
  newModel.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  newModel.add(tf.layers.flatten());
  newModel.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  const optimizer = tf.train.adam(0.01);
  newModel.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  // Warmup the new model.
  tf.tidy(() => newModel.predict(tf.zeros([1, ...INPUT_SHAPE])));
}

app();
