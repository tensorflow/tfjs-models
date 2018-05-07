import * as mobilenet from '../index';

async function page() {
  const net = await mobilenet.load();

  const img = document.getElementById('cat') as HTMLImageElement;
  const preds = await net.classify(img);

  console.log(preds);
}

page();
