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
  sentences.forEach(sentence => {
    const sentenceDom = document.createElement('div');
    sentenceDom.textContent = sentence;
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

  const selfSimilarity = [];
  for (let i = 0; i < sentences.length; i++) {
    selfSimilarity.push([]);
    for (let j = i; j < sentences.length; j++) {
      let score = dot(embeddingsArr[i], embeddingsArr[j]);
      selfSimilarity[i].push(score);
    }
  }

  const matrixSize = 250;
  const cellSize = matrixSize / sentences.length;
  const canvas = document.querySelector('canvas');
  canvas.width = matrixSize;
  canvas.height = matrixSize;

  const ctx = canvas.getContext('2d');

  for (let i = 0; i < sentences.length; i++) {
    for (let j = 0; j < sentences.length; j++) {
      let val = selfSimilarity[i][j];

      ctx.fillStyle = interpolateReds(val);
      ctx.fillRect(
          i * cellSize + j * cellSize, i * cellSize, cellSize, cellSize);
      ctx.fillRect(
          i * cellSize, i * cellSize + j * cellSize, cellSize, cellSize);
    }
  }
};

init();
