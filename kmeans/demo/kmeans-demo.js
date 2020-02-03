import {kMeans} from '@tensorflow-models/kmeans';
import {genRandomSamples} from '../src/util';
import * as tf from '@tensorflow/tfjs';

const nClusters = 4;
const nFeatures = 2;
const nSamplesPerCluster = 200;
const chartConfig = {
  pointColors: ['red', 'orange', 'blue', 'green'],
};

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
      legend: {
        align: 'end',
        labels: {
          usePointStyle: true,
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
  return chart;
}

function plotClusters(chart, samplesArr, centroidsArr, config = chartConfig) {
  const samplesDataset = {
    data: convertTensorArrayToChartData(samplesArr, nFeatures),
    radius: 2,
    label: 'Data',
  };
  const centroidsDataset = {
    data: convertTensorArrayToChartData(centroidsArr, nFeatures),
    pointBorderColor: 'black',
    borderWidth: 3,
    pointStyle: 'cross',
    pointRadius: 9,
    label: 'Centroids',
  };

  chart.data.datasets = [samplesDataset, centroidsDataset];
  chart.options.elements.point.backgroundColor = context => {
    const clusterId = Math.floor(context.dataIndex / nSamplesPerCluster);
    return config.pointColors[clusterId];
  };
  chart.update();
}

function updateClusters(
  chart,
  predictedArr,
  centroidsArr,
  config = chartConfig
) {
  const predictedCentroidsData = convertTensorArrayToChartData(
    centroidsArr,
    nFeatures
  );

  if (chart.data.datasets.length === 2) {
    chart.data.datasets.push({
      data: predictedCentroidsData,
      borderWidth: 3,
      pointStyle: 'cross',
      pointRadius: 9,
      label: 'Predicted centroids',
      pointBorderColor: context => config.pointColors[context.dataIndex],
    });
  } else {
    chart.data.datasets[
      chart.data.datasets.length - 1
    ].data = predictedCentroidsData;
  }

  chart.options.elements.point.backgroundColor = context => {
    const clusterId = predictedArr[context.dataIndex];
    return config.pointColors[clusterId];
  };
  chart.update();
}

async function onFit(model, samples, chart) {
  const predictions = model.fitPredict(samples);
  const predictionsArr = await predictions.data();
  const centroidsArr = await model.clusterCenters.data();
  tf.dispose(predictions);
  updateClusters(chart, predictionsArr, centroidsArr);
}

async function onFitOneCycle(model, samples, chart) {
  const predictions = model.fitOneCycle(samples);
  const predictionsArr = await predictions.data();
  const centroidsArr = await model.clusterCenters.data();
  tf.dispose(predictions);
  updateClusters(chart, predictionsArr, centroidsArr);
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
  const regenButton = document.getElementById('gen-train');
  regenButton.addEventListener('click', async () => {
    data = await onRegenData(chart);
  });

  const fitButton = document.getElementById('fit');
  fitButton.addEventListener('click', () => {
    const {samples} = data;
    onFit(model, samples, chart);
  });

  const fitOneButton = document.getElementById('fit-one');
  fitOneButton.addEventListener('click', () => {
    const {samples} = data;
    onFitOneCycle(model, samples, chart);
  });
}

onPageLoad();
