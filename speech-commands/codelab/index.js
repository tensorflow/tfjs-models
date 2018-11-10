'use strict';

let newModel;
let recognizer;
let activations = [];
let labels = [];

function save() {
  const activationsJson = activations.map(activation => {
    return {
      shape: activation.shape,
      values: Array.from(activation.dataSync())
    }
  });
  sessionStorage.setItem('activations', JSON.stringify(activationsJson));
  sessionStorage.setItem('labels', JSON.stringify(labels));
  console.log(`Saved ${activations.length} activations.`);
}
function load() {
  const activationsJson = JSON.parse(sessionStorage.getItem('activations'));
  activations = activationsJson.map(activationJson => {
    return tf.tensor(activationJson.values, activationJson.shape);
  });
  labels = JSON.parse(sessionStorage.getItem('labels'));
  console.log(`Loaded ${activations.length} activations.`);
}

// function normalize(x) {
//   const mean = -100;
//   const std = 22;
//   return x.map(x => (x - mean) / std);
// }

const NUM_FRAMES = 3;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];

function collect(label) {
  if (label == null) {
    return recognizer.stopListening();
  }
  recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
    const vals = data.subarray(-frameSize * NUM_FRAMES);
    //const {mean, variance} = tf.moments(vals);
    //console.log('mean', mean.get(), '\tvariance', variance.get());
    activations.push(tf.tensor(vals, [1, ...INPUT_SHAPE]));
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
  const xs = tf.concat(activations);

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

let delta = 0.1;

async function moveSlider(labelTensor) {
  const label = (await labelTensor.data())[0];
  if (label == 2) {
    return;
  }
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
    const vals = data.subarray(-frameSize * NUM_FRAMES);
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
