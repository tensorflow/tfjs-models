'use strict';

const NUM_FRAMES = 3;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

let model;
let recognizer;
let examples = [];

function save() {
  const jsonExamples =
      examples.map(e => ({label: e.label, vals: Array.from(e.vals)}));
  sessionStorage.setItem('examples', JSON.stringify(jsonExamples));
  console.log(`Saved ${examples.length} examples.`);
}
function load() {
  examples = JSON.parse(sessionStorage.getItem('examples'))
    .map(e => ({label: e.label, vals: new Float32Array(e.vals)}));
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
    examples.push({vals, label});
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
  model.add(tf.layers.depthwiseConv2d(
    {depthMultiplier: 8, kernelSize: [NUM_FRAMES, 3], activation: 'relu', inputShape: INPUT_SHAPE}));
  model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  // newModel.add(tf.layers.depthwiseConv2d(
  //     {depthMultiplier: 2, kernelSize: [1, 3], activation: 'relu'}));
  // newModel.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  // newModel.add(tf.layers.depthwiseConv2d(
  //     {depthMultiplier: 2, kernelSize: [1, 3], activation: 'relu'}));
  // newModel.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  // Warmup the new model.
  tf.tidy(() => model.predict(tf.zeros([1, ...INPUT_SHAPE])));
}

function setupUI() {
  document.getElementById('up').onmousedown = () => collect(0);
  document.getElementById('up').onmouseup = () => collect(null);

  document.getElementById('down').onmousedown = () => collect(1);
  document.getElementById('down').onmouseup = () => collect(null);

  document.getElementById('noise').onmousedown = () => collect(2);
  document.getElementById('noise').onmouseup = () => collect(null);

  document.getElementById('train').onmousedown = () => train();
  document.getElementById('listen').onmouseup = () => listen();
}

async function app() {
  // Load the model.
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  // Warmup.
  await recognizer.recognize(null);

  load();
  setupUI();
  buildModel();
}

app();
