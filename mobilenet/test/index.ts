import mobilenet from '../mobilenet';

async function page() {
  const net = await mobilenet(1, 0.25);

  const img = document.getElementById('cat') as HTMLImageElement;
  const preds = await net.classify(img);

  console.log(preds);
}

page();
