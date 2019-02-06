import * as use from '@tensorflow-models/universal-sentence-encoder';
import {interpolateReds} from 'd3-scale-chromatic';

const dot = (v1, v2) => {
  let product = 0;
  for (let i = 0; i < v1.length; i++) {
    product += v1[i] * v2[i];
  }
  return product;
};

const sentences = [
  'I like my phone.', 'Your cellphone looks great.', 'How old are you?',
  'What is your age?', 'An apple a day, keeps the doctors away.',
  'Eating strawberries is healthy.'
];

const renderSentences = () => {
  sentences.forEach((sentence, i) => {
    const sentenceDom = document.createElement('div');
    sentenceDom.textContent = `${i + 1}) ${sentence}`;
    document.querySelector('#sentences-container').appendChild(sentenceDom);
  });
};

const init = async () => {
  const model = await use.load();

  renderSentences();
  document.querySelector('#loading').style.display = 'none';

  const embeddings = await model.embed(sentences);
  const embeddingsData = embeddings.dataSync();
  const embeddingsArr = [];
  for (let i = 0; i < sentences.length; i++) {
    embeddingsArr.push(embeddingsData.slice(i * 512, i * 512 + 512));
  }

  const matrixSize = 250;
  const cellSize = matrixSize / sentences.length;
  const canvas = document.querySelector('canvas');
  canvas.width = matrixSize;
  canvas.height = matrixSize;

  const ctx = canvas.getContext('2d');

  const xLabelsContainer = document.querySelector('.x-axis');
  const yLabelsContainer = document.querySelector('.y-axis');

  for (let i = 0; i < sentences.length; i++) {
    const labelXDom = document.createElement('div');
    const labelYDom = document.createElement('div');

    labelXDom.textContent = i + 1;
    labelYDom.textContent = i + 1;
    labelXDom.style.left = (i * cellSize + cellSize / 2) + 'px';
    labelYDom.style.top = (i * cellSize + cellSize / 2) + 'px';

    xLabelsContainer.appendChild(labelXDom);
    yLabelsContainer.appendChild(labelYDom);

    for (let j = i; j < sentences.length; j++) {
      let score = dot(embeddingsArr[i], embeddingsArr[j]);

      ctx.fillStyle = interpolateReds(score);
      ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      ctx.fillRect(i * cellSize, j * cellSize, cellSize, cellSize);
    }
  }
};

init();
