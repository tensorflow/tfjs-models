import {kMeans} from '@tensorflow-models/kmeans';
import {genRandomSamples} from '../src/util';
import * as tf from '@tensorflow/tfjs';

const nClusters = 4;
const nFeatures = 2;
const nSamplesPerCluster = 200;
const chartConfig = {backgroundColors: ['red', 'yellow', 'blue', 'green']};

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

function initChart() {
  const ctx = document.getElementById('myChart').getContext('2d');
  const chart = new Chart(ctx, {
    type: 'scatter',
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
  return chart;
}

function plotClusters(chart, samplesArr, centroidsArr, config = chartConfig) {
  const samplesDataset = {
    data: convertTensorArrayToChartData(samplesArr, nFeatures),
    radius: 2,
  };
  const centroidsDataset = {
    data: convertTensorArrayToChartData(centroidsArr, nFeatures),
    borderWidth: 3,
    pointStyle: 'cross',
    pointRadius: 9,
    pointBorderColor: 'black',
  };

  chart.data.datasets = [samplesDataset, centroidsDataset];
  chart.options.elements.point.backgroundColor = context => {
    const clusterId = Math.floor(context.dataIndex / nSamplesPerCluster);
    return config.backgroundColors[clusterId];
  };
  chart.update();
}

function updateClusters(chart, predictedArr, config = chartConfig) {
  chart.options.elements.point.backgroundColor = context => {
    const clusterId = predictedArr[context.dataIndex];
    return config.backgroundColors[clusterId];
  };
  chart.update();
}

async function onFitButtonClick(model, samples, chart) {
  const predictions = await model.fitPredict(samples);
  const predictionsArr = await predictions.data();

  updateClusters(chart, predictionsArr);
}

async function onRegenData(chart) {
  const {centroids, samples} = genRandomSamples(
    nClusters,
    nSamplesPerCluster,
    nFeatures
  );
  const samplesArr = await samples.data();
  const centroidsArr = await centroids.data();

  plotClusters(chart, samplesArr, centroidsArr);
  return {centroids, samples};
}

async function onPageLoad() {
  // create model
  let model = kMeans({nClusters});

  // plot initial data
  const chart = initChart();
  let data = await onRegenData(chart);

  // set up event listeners
  const regenButton = document.getElementById('regen');
  regenButton.addEventListener('click', async () => {
    data = await onRegenData(chart);
  });

  const fitButton = document.getElementById('fit');
  fitButton.addEventListener('click', () => {
    const {samples} = data;
    onFitButtonClick(model, samples, chart);
  });
}

onPageLoad();
