/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licnses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import dat from 'dat.gui';
import * as tf from '@tensorflow/tfjs';
import * as posenet from '@tensorflow-models/posenet';
import { drawKeypoints, drawPoint, drawSegment, drawSkeleton,
  childToParentEdges, parentToChildEdges, renderImageToCanvas } from './demo_util';

const images = [
    'frisbee.jpg',
    'frisbee_2.jpg',
    'backpackman.jpg',
    'boy_doughnut.jpg',
    'soccer.png',
    'with_computer.jpg',
    'snowboard.jpg',
    'person_bench.jpg',
    'skiing.jpg',
    'fire_hydrant.jpg',
    'kyte.jpg',
    'looking_at_computer.jpg',
    'tennis.jpg',
    'tennis_standing.jpg',
    'truck.jpg',
    'on_bus.jpg',
    'tie_with_beer.jpg',
    'baseball.jpg',
    'multi_skiing.jpg',
    'riding_elephant.jpg',
    'skate_park_venice.jpg',
    'skate_park.jpg',
    'tennis_in_crowd.jpg',
    'two_on_bench.jpg',
];

/**
 * Draws a pose if it passes a minimum confidence onto a canvas.
 * Only the pose's keypoints that pass a minPartConfidence are drawn.
 */
function drawResults(canvas, poses,
    minPartConfidence, minPoseConfidence) {
    renderImageToCanvas(image, [513, 513], canvas);
    poses.forEach((pose) => {
        if (pose.score >= minPoseConfidence) {
            if (guiState.visualizeOutputs.showKeypoints)
              drawKeypoints(pose.keypoints,
                  minPartConfidence, canvas.getContext('2d'));

            if (guiState.visualizeOutputs.showSkeleton)
              drawSkeleton(pose.keypoints,
                  minPartConfidence, canvas.getContext('2d'));
       }
    });
}

const imageBucket = 'https://storage.googleapis.com/tfjs-models/assets/posenet/';

async function loadImage(imagePath) {
    const image = new Image();
    const promise = new Promise((resolve, reject) => {
        image.crossOrigin = '';
        image.onload = () => {
            resolve(image);
        };
    });

    image.src = `${imageBucket}${imagePath}`;
    return promise;
}

function singlePersonCanvas() {
    return document.querySelector('#single canvas');
}

function multiPersonCanvas() {
    return document.querySelector('#multi canvas');
}

/**
 * Draw the results from the single-pose estimation on to a canvas
 */
function drawSinglePoseResults(pose) {
    const canvas = singlePersonCanvas();
    drawResults(canvas, [pose],
        guiState.singlePoseDetection.minPartConfidence,
        guiState.singlePoseDetection.minPoseConfidence);

    const { part, showHeatmap, showOffsets } = guiState.visualizeOutputs;
    // displacements not used for single pose decoding
    const showDisplacements = false;
    const partId = Number(part);

    visualizeOutputs(partId, showHeatmap, showOffsets, showDisplacements, canvas.getContext('2d'));
}

/**
 * Draw the results from the multi-pose estimation on to a canvas
 */
function drawMultiplePosesResults(poses) {
    const canvas = multiPersonCanvas();
    drawResults(canvas, poses,
        guiState.multiPoseDetection.minPartConfidence,
        guiState.multiPoseDetection.minPoseConfidence);

    const { part, showHeatmap, showOffsets, showDisplacements  } = guiState.visualizeOutputs;
    const partId = Number(part);

    visualizeOutputs(partId, showHeatmap, showOffsets, showDisplacements, canvas.getContext('2d'));

}

/**
 *
 * @param partId The id of the part to draw the heatmap for
 *
 * @param drawOffsetVectors If the offset vectors should be drawn as well.
 */
function visualizeOutputs(partId, drawHeatmaps = true, drawOffsetVectors = true, drawDisplacements = true, ctx) {
  const { heatmapScores, offsets, displacementFwd, displacementBwd } = modelOutputs;
  const outputStride = Number(guiState.outputStride);

  const [ height, width ] = heatmapScores.shape;

  ctx.globalAlpha = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const score = heatmapScores.get(y, x, partId);

      // to save on performance, don't draw anything with a low score.
      if (score < 0.05) continue;

      // set opacity of drawn elements based on the score
      ctx.globalAlpha = score;

      if (drawHeatmaps)
        drawPoint(ctx, y * outputStride, x * outputStride, 2, 'yellow');

      const offsetsVectorY = offsets.get(y, x, partId);
      const offsetsVectorX = offsets.get(y, x, partId + 17);

      if (drawOffsetVectors) {
        drawSegment(
          [y * outputStride, x * outputStride],
          [y * outputStride + offsetsVectorY, x * outputStride + offsetsVectorX],
          'red',
          1.,
          ctx
        );
      }

      if (drawDisplacements) {
        ctx.globalAlpha *= score;
        const numEdges = displacementFwd.shape[2] / 2;

        const offsetX = x * outputStride + offsetsVectorX;
        const offsetY = y * outputStride + offsetsVectorY;

        const forwardEdgeIds = parentToChildEdges[partId] || [];

        if (forwardEdgeIds.length > 0) {
          forwardEdgeIds.forEach(forwardEdgeId => {
            const forwardY = displacementFwd.get(y, x, forwardEdgeId);
            const forwardX = displacementFwd.get(y, x, forwardEdgeId + numEdges);

            drawSegment([offsetY, offsetX], [offsetY + forwardY, offsetX + forwardX], 'blue', 1., ctx);
          });
        }

        const backwardsEdgeIds = childToParentEdges[partId] || [];

        backwardsEdgeIds.forEach(backwardsEdgeId => {
          if (typeof backwardsEdgeId !== 'undefined') {
            const backwardY= displacementBwd.get(y, x, backwardsEdgeId);
            const backwardX = displacementBwd.get(y, x, backwardsEdgeId + numEdges);

            drawSegment([offsetY, offsetX], [offsetY + backwardY, offsetX + backwardX], 'blue', 1., ctx);
          }
        })
      }
    }

    ctx.globalAlpha = 1;
  }
}

/**
 * Converts the raw model output results into single-pose estimation results
 */
async function decodeSinglePoseAndDrawResults() {
    if (!modelOutputs) {
        return;
    }

    const pose = await posenet.decodeSinglePose(
        modelOutputs.heatmapScores, modelOutputs.offsets,
        guiState.outputStride);

    drawSinglePoseResults(pose);
}

/**
 * Converts the raw model output results into multi-pose estimation results
 */
async function decodeMultiplePosesAndDrawResults() {
    if (!modelOutputs) {
        return;
    }

    const poses = await posenet.decodeMultiplePoses(
        modelOutputs.heatmapScores, modelOutputs.offsets,
        modelOutputs.displacementFwd, modelOutputs.displacementBwd,
        guiState.outputStride,
        guiState.multiPoseDetection.maxDetections, guiState.multiPoseDetection);

    drawMultiplePosesResults(poses);
}

function decodeSingleAndMultiplePoses() {
    decodeSinglePoseAndDrawResults();
    decodeMultiplePosesAndDrawResults();
}

function setStatusText(text) {
    const resultElement = document.getElementById('status');
    resultElement.innerText = text;
}

let image = null;
let modelOutputs = null;

/**
 * Purges variables and frees up GPU memory using dispose() method
 */
function disposeModelOutputs() {
    if (modelOutputs) {
        modelOutputs.heatmapScores.dispose();
        modelOutputs.offsets.dispose();
        modelOutputs.displacementFwd.dispose();
        modelOutputs.displacementBwd.dispose();
    }
}

/**
 * Loads an image, feeds it into posenet the posenet model, and
 * calculates poses based on the model outputs
 */
async function testImageAndEstimatePoses(net) {
    setStatusText('Predicting...');
    document.getElementById('results').style.display = 'none';

    // Purge prevoius variables and free up GPU memory
    disposeModelOutputs();

    // Load an example image
    image = await loadImage(guiState.image);

    // Creates a tensor from an image
    const input = tf.fromPixels(image);

    // Stores the raw model outputs from both single- and multi-pose resutls can
    // be decoded
    modelOutputs = await net.predictForMultiPose(input, guiState.outputStride);

    // Process the model outputs to convert into poses
    await decodeSingleAndMultiplePoses();

    setStatusText('');
    document.getElementById('results').style.display = 'block';
    input.dispose();
}

let guiState;

function setupGui(net) {
    guiState = {
        outputStride: 16,
        image: 'tennis_in_crowd.jpg',
        detectPoseButton: () => {
            testImageAndEstimatePoses(
                net);
        },
        singlePoseDetection: {
            minPartConfidence: 0.5,
            minPoseConfidence: 0.5,
        },
        multiPoseDetection: {
            minPartConfidence: 0.5,
            minPoseConfidence: 0.5,
            scoreThreshold: 0.5,
            nmsRadius: 20.0,
            maxDetections: 15,
        },
        visualizeOutputs: {
          showKeypoints: true,
          showSkeleton: true,
          part: 0,
          showHeatmap: false,
          showOffsets: false,
          showDisplacements: false
        }
    };

    const gui = new dat.GUI();
    // Output stride:  Internally, this parameter affects the height and width of the layers
    // in the neural network. The lower the value of the output stride the higher the accuracy
    // but slower the speed, the higher the value the faster the speed but lower the accuracy.
    gui.add(guiState, 'outputStride', [8, 16, 32])
        .onChange((outputStride) => {
           guiState.outputStride = Number(outputStride);
           testImageAndEstimatePoses(net);
        });
    gui.add(guiState, 'image', images)
        .onChange(() => testImageAndEstimatePoses(net));

    // Pose confidence: the overall confidence in the estimation of a person's
    // pose (i.e. a person detected in a frame)
    // Min part confidence: the confidence that a particular estimated keypoint
    // position is accurate (i.e. the elbow's position)

    const multiPoseDetection = gui.addFolder('Multi Pose Estimation');
    multiPoseDetection.open();
    multiPoseDetection.add(
            guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0)
        .onChange(decodeMultiplePosesAndDrawResults);
    multiPoseDetection.add(
            guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0)
        .onChange(decodeMultiplePosesAndDrawResults);

    // nms Radius: controls the minimum distance between poses that are returned
    // defaults to 20, which is probably fine for most use cases
    multiPoseDetection.add(guiState.multiPoseDetection, 'nmsRadius', 0.0, 40.0)
        .onChange(decodeMultiplePosesAndDrawResults);
    multiPoseDetection.add(guiState.multiPoseDetection, 'maxDetections')
        .min(1)
        .max(20)
        .step(1)
        .onChange(decodeMultiplePosesAndDrawResults);

    const singlePoseDetection = gui.addFolder('Single Pose Estimation');
    singlePoseDetection.add(
            guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0)
        .onChange(decodeSinglePoseAndDrawResults);
    singlePoseDetection.add(
            guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0)
        .onChange(decodeSinglePoseAndDrawResults);
    singlePoseDetection.open();

    const visualizeOutputs = gui.addFolder('Visualize Outputs');

    visualizeOutputs.add(guiState.visualizeOutputs, 'showKeypoints')
      .onChange(decodeSingleAndMultiplePoses);
    visualizeOutputs.add(guiState.visualizeOutputs, 'showSkeleton')
      .onChange(decodeSingleAndMultiplePoses);
    visualizeOutputs.add(
      guiState.visualizeOutputs, 'part', posenet.partIds)
      .onChange(decodeSingleAndMultiplePoses);
    visualizeOutputs.add(
      guiState.visualizeOutputs, 'showHeatmap')
      .onChange(decodeSingleAndMultiplePoses);
    visualizeOutputs.add(
      guiState.visualizeOutputs, 'showOffsets')
      .onChange(decodeSingleAndMultiplePoses);
    visualizeOutputs.add(
      guiState.visualizeOutputs, 'showDisplacements')
      .onChange(decodeSingleAndMultiplePoses);




    visualizeOutputs.open();
}

/**
 * Kicks off the demo by loading the posenet model and estimating
 * poses on a default image
 */
export async function bindPage() {
    const net = await posenet.load();

    setupGui(net);

    await testImageAndEstimatePoses(net);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('main').style.display = 'block';
}

bindPage();
