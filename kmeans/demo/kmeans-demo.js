import {kMeans, KMeansArgs, KMeansClustering} from '@tensorflow-models/kmeans';
import {genRandomSamples} from '../src/util';
import * as tf from '@tensorflow/tfjs';

const nClusters = 4;
const nFeatures = 2;
const nSamplesPerCluster = 200;
const chartConfig = {
  pointColors: [
    [255, 0, 0],
    [255, 180, 0],
    [0, 0, 255],
    [32, 200, 32],
  ],
  scalesOptions: {
    xAxes: [
      {
        type: 'linear',
        position: 'bottom',
      },
    ],
  },
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
      scales: chartConfig.scalesOptions,
    },
  });
  return chart;
}

function plotClusters(chart, samplesArr, centroidsArr, config = chartConfig) {
  const samplesDataset = {
    data: convertTensorArrayToChartData(samplesArr, nFeatures),
    radius: 2,
    label: 'Data',
    fill: false,
  };
  const centroidsDataset = {
    data: convertTensorArrayToChartData(centroidsArr, nFeatures),
    pointBorderColor: 'black',
    borderWidth: 3,
    pointStyle: 'cross',
    pointRadius: 9,
    label: 'Centroids',
  };

  // chart.data.datasets contains [<train/test instances>, <real centroids>, <predicted centroids>]
  chart.data.datasets = [samplesDataset, centroidsDataset];
  chart.data.datasets[0].backgroundColor = context => {
    const clusterId = Math.floor(context.dataIndex / nSamplesPerCluster);
    const [r, g, b] = config.pointColors[clusterId];
    return `rgb(${r},${g},${b},0.5)`;
  };
  // resume auto scaling
  chart.options.scales = chartConfig.scalesOptions;
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

  // chart.data.datasets contains [<train/test instances>, <real centroids>, <predicted centroids>]
  if (chart.data.datasets.length === 2) {
    chart.data.datasets.push({
      data: predictedCentroidsData,
      borderWidth: 3,
      pointStyle: 'cross',
      pointRadius: 9,
      label: 'Predicted centroids',
      pointBorderColor: context => {
        const [r, g, b] = config.pointColors[context.dataIndex];
        return `rgb(${r},${g},${b})`;
      },
    });
  } else {
    chart.data.datasets[
      chart.data.datasets.length - 1
    ].data = predictedCentroidsData;
  }

  chart.data.datasets[0].backgroundColor = context => {
    const clusterId = predictedArr[context.dataIndex];
    const [r, g, b] = config.pointColors[clusterId];
    return `rgb(${r},${g},${b},0.5)`;
  };
  chart.update();
}

function plotTestData(chart, samplesArr, predictedArr, config = chartConfig) {
  // chart.data.datasets contains [<train/test instances>, <real centroids>, <predicted centroids>]
  chart.data.datasets[0].data = convertTensorArrayToChartData(
    samplesArr,
    nFeatures
  );

  chart.data.datasets[0].backgroundColor = context => {
    const clusterId = predictedArr[context.dataIndex];
    const [r, g, b] = config.pointColors[clusterId];
    return `rgb(${r},${g},${b},0.5)`;
  };

  // prevent auto-scaling
  chart.options.scales.xAxes[0].ticks.min = chart.scales['x-axis-1'].min;
  chart.options.scales.xAxes[0].ticks.max = chart.scales['x-axis-1'].max;
  chart.options.scales.yAxes[0].ticks.min = chart.scales['y-axis-1'].min;
  chart.options.scales.yAxes[0].ticks.max = chart.scales['y-axis-1'].max;

  chart.update();
}

async function onFit(model, samples, chart) {
  const predictionsArr = await model.fitPredict(samples);
  const centroidsArr = await model.clusterCenters.data();
  updateClusters(chart, predictionsArr, centroidsArr);
}

async function onFitOneCycle(model, samples, chart) {
  const predictionsArr = await model.fitOneCycle(samples);
  const centroidsArr = await model.clusterCenters.data();
  console.log(centroidsArr);
  updateClusters(chart, predictionsArr, centroidsArr);
}

async function onRegenTrainData(chart) {
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

async function onRegenTestData(chart, model) {
  const testDataConfig = {
    nClusters: 1,
    nSamplesPerCluster: 500,
    nFeatures: nFeatures,
    variance: 3,
    embiggenFactor: 2,
  };
  const {samples} = await genRandomSamples(
    testDataConfig.nClusters,
    testDataConfig.nSamplesPerCluster,
    testDataConfig.nFeatures,
    testDataConfig.variance,
    testDataConfig.embiggenFactor
  );
  const predictionsArr = await model.predict(samples);
  const samplesArr = await samples.data();
  tf.dispose(samples);

  plotTestData(chart, samplesArr, predictionsArr);
}

async function onPageLoad() {
  // create model
  let model = kMeans({nClusters});

  // plot initial data
  const chart = initChart();
  let trainData = await onRegenTrainData(chart);

  // set up event listeners
  const genTrainButton = document.getElementById('gen-train');
  const fitButton = document.getElementById('fit');
  const fitOneButton = document.getElementById('fit-one');
  const genTestButton = document.getElementById('gen-test');

  genTrainButton.addEventListener('click', async () => {
    trainData = await onRegenTrainData(chart);
    genTestButton.disabled = true;
  });

  fitButton.addEventListener('click', async () => {
    const {samples} = trainData;
    await onFit(model, samples, chart);
    genTestButton.disabled = false;
  });

  fitOneButton.addEventListener('click', async () => {
    const {samples} = trainData;
    await onFitOneCycle(model, samples, chart);
    genTestButton.disabled = false;
  });

  genTestButton.addEventListener('click', async () => {
    await onRegenTestData(chart, model);
  });
}

onPageLoad();
