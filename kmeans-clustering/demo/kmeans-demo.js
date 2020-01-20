import {kmeans} from '@tensorflow-models/kmeans';
import {genRandomSamples} from '../src/util';
import * as tf from '@tensorflow/tfjs';

const nClusters = 4;
const nFeatures = 2;
let model;

async function prepareData(nSamplesPerCluster) {
  const {centroids, samples} = genRandomSamples(nClusters, nSamplesPerCluster, nFeatures);
  const allSamples = await samples.data();
  const allCentroids = await centroids.data();
  const samplesArr = [];

  const nElePerSample = nFeatures * nSamplesPerCluster;
  for (let i = 0; i < nClusters; i++) {
    samplesArr.push(allSamples.slice(i * nElePerSample, (i + 1) * nElePerSample));
  }
  return {samplesArr, allCentroids};
}

function convertTensorArrayToChartData(arr, nDims) {
  const fieldNames = ['x', 'y', 'z', 't', 'u', 'v'];
  const res = [];
  arr.forEach((ele, i) => {
    const dim = i % nDims;
    if (dim === 0) {
      res.push({});
    }
    res[res.length - 1][fieldNames[dim]] = ele;
  });
  return res;
}

function plotClusters(samplesArr, centroids, nSamplesPerCluster) {
  const ctx = document.getElementById('myChart').getContext('2d');
  const backgroundColors = ['red', 'yellow', 'blue', 'green'];

  const datasets = samplesArr.map((arr, i) => ({
    label: `Cluster ${i}`,
    data: convertTensorArrayToChartData(arr, nFeatures),
    backgroundColor: backgroundColors[i],
  }));
  const myChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets,
    },
    options: {
      scales: {
        xAxes: [
          {
            type: 'linear',
            position: 'bottom',
          },
        ],
      },
    },
  });
}

function fitData(data) {
  model.fitPredict(data);
}

async function onPageLoad() {
  // create model
  model = kmeans({nClusters});

  // plot initial data
  const nSamplesPerCluster = 200;
  const {samplesArr, allCentroids} = await prepareData(nSamplesPerCluster);
  plotClusters(samplesArr, allCentroids, nSamplesPerCluster);

  // set up event listener
  const fitButton = document.getElementById('fit');
  fitButton.addEventListener('click', () => {
    fitData(tf.tensor2d(samplesArr, [nSamplesPerCluster, nFeatures]));
  });
}

onPageLoad();
