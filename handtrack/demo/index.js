import "./anchors.js";
import "./handdetector.js";
import "./pipeline.js";

const color = 'red';

/**
 * Draw pose keypoints onto a canvas
 */
function drawKeypoints(ctx, keypoints) {
  let keypointsArray = keypoints.arraySync();

  for (let i = 0; i < keypointsArray.length; i++) {
    const y = keypointsArray[i][0];
    const x = keypointsArray[i][1];
    drawPoint(ctx, x, y, 3, color);
  }
}

function drawBoundingBox(ctx, bbox) {
  let startEnd = bbox.startEndTensor.arraySync()[0];
  let x = startEnd[0];
  let y = startEnd[1];
  let width = startEnd[2] - startEnd[0];
  let height = startEnd[3] - startEnd[1];
  ctx.strokeRect(x, y, width, height);
}

function drawPoint(ctx, y, x, r, color) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = color;
  ctx.fill();
}

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

function isDesktop() {
  return !isMobile();
}

function isChrome() {
  return true;
}

function isSupportedPlatform() {
  return true; // isAndroid() || isiOS() || (isChrome() && isDesktop());
}

// Model loading
const HANDDETECT_MODEL_PATH = "./handdetector_hourglass_short_2019_03_25_v0-web/model.json";
const HANDTRACK_MODEL_PATH = "./handskeleton_3d_handflag_2019_08_19_v0-web/model.json"

tf.registerOp('Prelu', (node) => {
  const x = node.inputs[0];
  const alpha = node.inputs[1];
  return tf.prelu(x, alpha);
});

function loadHandDetect() {
  return tf.loadGraphModel(HANDDETECT_MODEL_PATH);
}

function loadHandTrack() {
  return tf.loadGraphModel(HANDTRACK_MODEL_PATH);
}

const statusElement = document.getElementById("status");
const status = msg => statusElement.innerText = msg;

async function setupCamera() {
  if (!isSupportedPlatform()) {
    throw new Error("Your browser doesn't supported yet.");
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();
  return video;
}

/**
 * Start the demo.
 */
const bindPage = async () => {
  const handtrackModel = await loadHandTrack();
  const handdetectModel = await loadHandDetect();
  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = e.message;
    info.style.display = 'block';
    throw e;
  }

  landmarksRealTime(video, handdetectModel, handtrackModel);
}


const landmarksRealTime = async (video, handdetectModel, handtrackModel) => {
  const stats = new Stats();
  stats.showPanel(0);
  document.body.appendChild(stats.dom);

  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  const canvas = document.getElementById('output');

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  const ctx = canvas.getContext('2d');

  video.width = videoWidth;
  video.height = videoHeight;

  ctx.clearRect(0, 0, videoWidth, videoHeight);
  ctx.strokeStyle = "red";

  const handdetect = new HandDetectModel(handdetectModel, 256, 256);

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  const pipeline = new HandPipeline(handdetect, handtrackModel);

  async function frameLandmarks() {
    stats.begin();
    tf.tidy(function () {
      const image = tf.browser.fromPixels(video).toFloat().expandDims(0);

      ctx.drawImage(video, 0, 0, videoWidth, videoHeight,
        0, 0, canvas.width, canvas.height);

      let meshes = pipeline.next_meshes(image);
      if (meshes) {
        drawKeypoints(ctx, meshes);
      }
    });
    stats.end();

    requestAnimationFrame(frameLandmarks);
  };

  frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

bindPage();
