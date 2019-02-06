import * as use from '@tensorflow-models/universal-sentence-encoder';

const sentences = [
  'I like my phone.', 'Your cellphone looks great.', 'Will it snow tomorrow?',
  'Recently a lot of hurricanes have hit the US.',
  'An apple a day, keeps the doctors away.', 'Eating strawberries is healthy.'
];

const init = async () => {
  const model = await use.load();

  document.querySelector('#loading').style.display = 'none';

  const embeddings = await model.embed(sentences);
  console.log(embeddings);
};

init();