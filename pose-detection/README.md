# Pose Detection

This package provides multiple state-of-the-art models for running real-time pose detection.

Currently, we provide 3 model options:

#### MoveNet
[Demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet)

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
It can run at 50+ fps on modern laptop and phones.

#### MediapipeBlazepose:
[Demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose)

MediapipeBlazepose can detect 33 keypoints, in addition to the 17 COCO keypoints,
it provides additional keypoints for face, hands and feet.

#### PoseNet
[Demo](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=posenet)

PoseNet can detect multiple poses, each pose contains 17 keypoints.

-------------------------------------------------------------------------------
## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Keypoint Diagram](#keypoint-diagram)
4. [Example Code and Demos](#example-code-and-demos)

-------------------------------------------------------------------------------

## Installation

Via script tags:

```html
<!-- Require the peer dependencies of pose-detection. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection"></script>
```

Via npm:

```sh
yarn add @tensorflow-models/pose-detection
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
```

-------------------------------------------------------------------------------

## Usage
If you are using via npm, you need to import the libraries first.
### Import the libraries:
```javascript
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

// Register one of the TF.js backends.
import '@tensorflow/tfjs-backend-webgl';
```

### Create a detector:
Choose a model from `poseDetection.SupportedModels` enum list and pass it to the
`createDetector` method.

`SupportedModels` include `MoveNet`, `MediapipeBlazepose` and `PoseNet`.

```javascript
const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet);
```

This method will fetch and load the model in memory. It
will also prepare the detector to be ready to run. There is a default setting
for each model, if you want more options, you can pass in a `ModelConfig` as the
second parameter. For details, see the README for the model you are interested in.

### Pose Estimation
Now you can use the detector to detect poses. The
`estimatePoses` method accepts both image and video in many formats, including:
`tf.Tensor3D`, `ImageData`, `HTMLVideoElement`, `HTMLImageElement`,
`HTMLCanvasElement`. If you want more options, you can pass in an
`EstimationConfig` as the second parameter.
For details, see the README for the model you are interested in.

```javascript
const poses = await detector.estimatePoses(image);
```

The returned `poses` list contains detected poses for each individual in the
image. For single-person models, there will only be one element in the list.
Currently, only PoseNet supports multi-pose estimation. If the model cannot
detect any people, the list will be empty.

For each pose, it contains a confidence score of the pose and an array of
keypoints. PoseNet and MoveNet both return 17 keypoints.
MediapipeBlazepose returns 33 keypoints. Each keypoint contains `x`, `y`,
`score` and `name`.

Example output:
```javascript
[
  {
    score: 0.8,
    keypoints: [
      {x: 230, y: 220, score: 0.9, name: "nose"},
      {x: 212, y: 190, score: 0.8, name: "left_eye"},
      ...
    ]
  }
]
```

`x` and `y` represent the actual keypoint position in the image. If you need
normalized keypoint positions, you can use the method
`poseDetection.calculator.keypointsToNormalizedKeypoints(keypoints, imageSize)`
to convert `x` and `y` to `[0, 1]` range.

The `score` ranges from `0` to `1`. It represents the model's confidence of a
keypoint. Usually, keypoints with low confidence scores should not be used.
Each application may require a custom confidence threshold. For applications
that require high precision, we recommend a larger confidence value. Conversely,
applications that require high recall may choose to lower the threshold. The
confidence values are not calibrated between models, and therefore setting a
proper confidence threshold may involve some experimentation.

The `name` provides a label for each keypoint, such as 'nose', 'left_eye',
'right_knee', etc.

-------------------------------------------------------------------------------

## Keypoint Diagram
See the diagram below for what those keypoints are and their index in the array.

### COCO Keypoints: Used in MoveNet and PoseNet
![COCO Keypoints](https://storage.googleapis.com/movenet/coco-keypoints-500.png)


0: nose  \
1: left_eye  \
2: right_eye  \
3: left_ear  \
4: right_ear  \
5: left_shoulder  \
6: right_shoulder  \
7: left_elbow  \
8: right_elbow  \
9: left_wrist  \
10: right_wrist  \
11: left_hip  \
12: right_hip  \
13: left_knee  \
14: right_knee  \
15: left_ankle  \
16: right_ankle

### Blazepose Keypoints: Used in Mediapipe Blazepose
![Blazepose Keypoints](https://storage.googleapis.com/mediapipe/blazepose-keypoints-500.png)


0: nose  \
1: right_eye_inner \
2: right_eye  \
3: right_eye_outer  \
4: left_eye_inner  \
5: left_eye  \
6: left_eye_outer  \
7: right_ear  \
8: left_ear  \
9: mouth_right  \
10: mouth_left  \
11: right_shoulder  \
12: left_shoulder  \
13: right_elbow  \
14: left_elbow  \
15: right_wrist  \
16: left_wrist  \
17: right_pinky  \
18: left_pinky  \
19: right_index  \
20: left_index  \
21: right_thumb  \
22: left_thumb  \
23: right_hip  \
24: left_hip  \
25: right_knee  \
26: left_knee  \
27: right_ankle  \
28: left_ankle  \
29: right_heel  \
30: left_heel  \
31: right_foot_index  \
32: left_foot_index

-------------------------------------------------------------------------------

## Example Code and Demos
You may reference the demos for code examples. Details for how to run the demos
are included in the `demo/`
[folder](https://github.com/tensorflow/tfjs-models/tree/master/pose-detection/demo).
