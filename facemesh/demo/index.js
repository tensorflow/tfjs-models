import * as faceMesh from '@tensorflow-models/facemesh';

let model, ctx, videoWidth, videoHeight, video, canvas;

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {facingMode: 'user'},
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

const renderPrediction =
    async () => {
  const prediction = await model.estimateFace(video);
  if (prediction) {
    const keypoints = prediction.mesh;

    ctx.drawImage(
        video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
        canvas.height);

    const r = 1;

    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  requestAnimationFrame(renderPrediction);
}

const setupPage =
    async () => {
  await setupCamera();
  video.play();

  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  model = await faceMesh.load();

  renderPrediction();
}

setupPage();