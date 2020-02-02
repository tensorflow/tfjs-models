import {kMeans} from '@tensorflow-models/kmeans';
import {genRandomSamples} from '../src/util';
import * as tf from '@tensorflow/tfjs';

const nClusters = 4;
const nFeatures = 2;
const nSamplesPerCluster = 200;
let samplesArr, centroidsArr, model, chart;
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

function plotClusters(config = chartConfig) {
  const ctx = document.getElementById('myChart').getContext('2d');
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

  chart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [samplesDataset, centroidsDataset],
    },
    options: {
      elements: {
        point: {
          backgroundColor: context => {
            const clusterId = Math.floor(
              context.dataIndex / nSamplesPerCluster
            );
            return config.backgroundColors[clusterId];
          },
        },
      },
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

function updateClusters(predictedArr, config = chartConfig) {
  chart.options.elements.point.backgroundColor = context => {
    const clusterId = predictedArr[context.dataIndex];
    return config.backgroundColors[clusterId];
  };
  chart.update();
}

async function onFitButtonClick(samples) {
  const predictions = await model.fitPredict(samples);
  const predictionsArr = await predictions.data();

  updateClusters(predictionsArr);
}

async function onPageLoad() {
  // create model
  model = kMeans({nClusters});

  // plot initial data
  const {centroids, samples} = genRandomSamples(
    nClusters,
    nSamplesPerCluster,
    nFeatures
  );
  samplesArr = await samples.data();
  centroidsArr = await centroids.data();
  plotClusters();

  // set up event listener
  const fitButton = document.getElementById('fit');
  fitButton.addEventListener('click', () => onFitButtonClick(samples));
}

onPageLoad();
