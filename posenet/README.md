# Pose Detection in the Browser: PoseNet Model

This package contains a standalone model called PoseNet, as well as some demos, for running real-time pose estimation in the browser using TensorFlow.js.

<img src="./demos/camera.gif" alt="cameraDemo" style="width: 600px;"/>

PoseNet can be used to estimate either a single pose or multiple poses, meaning there is a version of the algorithm that can detect only one person in an image/video and one version that can detect multiple persons in an image/video.

[Refer to this blog post](https://medium.com/p/1cf363e812ce) for a high-level description of PoseNet running on Tensorflow.js.

## Installation

You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/tfjs"></script>
<script src="https://unpkg.com/posenet"></script>
```

Or you can install it via npm for use in a TypeScript / ES6 project.

```sh
npm install @tensorflow-models/posenet
```

## Usage

Either a single pose our multiple poses can be estimated from an image.
Each methodology has its own algorithm and set of parameters.

### Keypoints

All keypoints are indexed by part id.  The parts and their ids are:

| Id | Part |
| -- | -- |
| 0 | nose |
| 1 | leftEye |
| 2 | rightEye |
| 3 | leftEar |
| 4 | rightEar |
| 5 | leftShoulder |
| 6 | rightShoulder |
| 7 | leftElbow |
| 8 | rightElbow |
| 9 | leftWrist |
| 10 | rightWrist |
| 11 | leftHip |
| 12 | rightHip |
| 13 | leftKnee |
| 14 | rightKnee |
| 15 | leftAnkle |
| 16 | rightAnkle |

### Single-Person Pose Estimation

Single pose estimation is the simpler and faster of the two algorithms. Its ideal use case is for when there is only one person in the image. The disadvantage is that if there are multiple persons in an image, keypoints from both persons will likely be estimated as being part of the same single pose—meaning, for example, that person #1’s left arm and person #2’s right knee might be conflated by the algorithm as belonging to the same pose.

```javascript
const pose = await poseNet.estimateSinglePose(image, inputSize, reverse, outputStride);
```

#### Inputs

* **image** - ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement
   The input image to feed through the network.
* **inputSize** - The size to scale the image to before feeding through the network. Defaults to 513.  Must have a value which when 1 is subtracted from it, is divisible by the output stride. Sample acceptable values are 29, 161, 193, 257, 289, 321, 353, 385, 417, 449, 481, 513. Set this number lower to scale down the image and increase the speed when feeding through the network.
* **reverse** - If the input image should be reversed horizontally.  Defaults to false. This is useful for video elements where the input image is usually reversed.
* **outputStride** - the desired stride for the outputs when feeding the image through the model.  Must be 32, 16, 8.  Defaults to 16.  The higher the number, the faster the performance but slower the accuracy, and visa versa.

#### Returns

It returns a `pose` with a confidence score and an array of keypoints indexed by part id, each with a score and position.

#### Example Usage

##### Estimating a single pose from an image

```javascript
import * as posenet from '@tensorflow-models/posenet';
const inputSize = 321;
const outputStride = 16;
const reverse = false;

async function estimatePoseOnImage(imageElement) {
  // load the posenet model from a checkpoint
  const net = await posenet.posenet();

  const pose = await net.estimateSinglePose(imageElement, inputSize, reverse, outputStride);

  return pose;
}

const imageElement = document.getElementById('cat');

const pose = estimatePoseOnImage(imageElement);

console.log(pose);

```

which would produce the output:

```json
{
  "score": 0.32371445304906,
  "keypoints": [
    {
      "position": {
        "y": 76.291801452637,
        "x": 253.36747741699
      },
      "part": "nose",
      "score": 0.99539834260941
    },
    {
      "position": {
        "y": 71.10383605957,
        "x": 253.54365539551
      },
      "part": "leftEye",
      "score": 0.98781454563141
    },
    {
      "position": {
        "y": 71.839515686035,
        "x": 246.00454711914
      },
      "part": "rightEye",
      "score": 0.99528175592422
    },
    {
      "position": {
        "y": 72.848854064941,
        "x": 263.08151245117
      },
      "part": "leftEar",
      "score": 0.84029853343964
    },
    {
      "position": {
        "y": 79.956565856934,
        "x": 234.26812744141
      },
      "part": "rightEar",
      "score": 0.92544466257095
    },
    {
      "position": {
        "y": 98.34538269043,
        "x": 399.64068603516
      },
      "part": "leftShoulder",
      "score": 0.99559044837952
    },
    {
      "position": {
        "y": 95.082359313965,
        "x": 458.21868896484
      },
      "part": "rightShoulder",
      "score": 0.99583911895752
    },
    {
      "position": {
        "y": 94.626205444336,
        "x": 163.94561767578
      },
      "part": "leftElbow",
      "score": 0.9518963098526
    },
    {
      "position": {
        "y": 150.2349395752,
        "x": 245.06030273438
      },
      "part": "rightElbow",
      "score": 0.98052614927292
    },
    {
      "position": {
        "y": 113.9603729248,
        "x": 393.19735717773
      },
      "part": "leftWrist",
      "score": 0.94009721279144
    },
    {
      "position": {
        "y": 186.47859191895,
        "x": 257.98034667969
      },
      "part": "rightWrist",
      "score": 0.98029226064682
    },
    {
      "position": {
        "y": 208.5266418457,
        "x": 284.46710205078
      },
      "part": "leftHip",
      "score": 0.97870296239853
    },
    {
      "position": {
        "y": 209.9910736084,
        "x": 243.31219482422
      },
      "part": "rightHip",
      "score": 0.97424703836441
    },
    {
      "position": {
        "y": 281.61965942383,
        "x": 310.93188476562
      },
      "part": "leftKnee",
      "score": 0.98368924856186
    },
    {
      "position": {
        "y": 282.80120849609,
        "x": 203.81164550781
      },
      "part": "rightKnee",
      "score": 0.96947449445724
    },
    {
      "position": {
        "y": 360.62716674805,
        "x": 292.21047973633
      },
      "part": "leftAnkle",
      "score": 0.8883239030838
    },
    {
      "position": {
        "y": 347.41177368164,
        "x": 203.88229370117
      },
      "part": "rightAnkle",
      "score": 0.8255187869072
    }
  ]
}
```

### Multi-Person Pose Estimation

Multiple Pose estimation can decode multiple poses in an image. It is more complex and slightly slower than the single pose-algorithm, but has the advantage that if multiple people appear in an image, their detected keypoints are less likely to be associated with the wrong pose. Even if the use case is to detect a single person’s pose, this algorithm may be more desirable in that the accidental effect of two poses being joined together won’t occur when multiple people appear in the image. It uses the `Fast greedy decoding` algorithm from the research paper [PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model](https://arxiv.org/pdf/1803.08225.pdf).

```javascript
const poses = await net.estimateMultiplePoses(image, inputSize, reverse, outputStride, maxPoseDetections, scoreThreshold, nmsRadius);
```

#### Inputs

* **image** - ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement
   The input image to feed through the network.
* **inputSize** - The size to scale the image to before feeding through the network. Defaults to 513.  Must have a value which when 1 is subtracted from it, is divisible by the output stride. Sample acceptable values are 29, 161, 193, 257, 289, 321, 353, 385, 417, 449, 481, 513. Set this number lower to scale down the image and increase the speed feeding when through the network.
* **reverse** - If the input image should be reversed horizontally.  Defaults to false. This is useful for video elements where the input image is usually reversed.
* **outputStride** - the desired stride for the outputs when feeding the image through the model.  Must be 32, 16, 8.  Defaults to 16.  The higher the number, the faster the performance but slower the accuracy, and visa versa.
* **maxPoseDetections** (optional) - the maximum number of poses to detect. Defaults to 5.
* **scoreThreshold** (optional) - Only return instance detections that have root part score greater or equal to this value. Defaults to 0.5.
* **nmsRadius** (optional) - Non-maximum suppression part distance. It needs to be strictly positive. Two parts suppress each other if they are less than `nmsRadius` pixels away. Defaults to 20.

#### Returns

It returns a `promise` that resolves with an array of `poses`, each with a confidence score and an array of `keypoints` indexed by part id, each with a score and position.

#### Example Usage

##### Estimating multiple poses from an image

```javascript
import * as posenet from '@tensorflow-models/posenet';

const inputSize = 321;
const outputStride = 16;
const reverse = false;
const outputStride = 16;
const maxPoseDetections = 2;

async function estimateMultiplePosesOnImage(imageElement) {
  const net = await posenet.posenet();

  // estimate poses
  const poses = await net.estimateMultiplePoses(imageElement,
    inputSize, reverse, outputStride, maxPoseDetections);

  return poses;
}

const imageElement = document.getElementById('people');

const poses = estimateMultiplePosesOnImage(imageElement);

console.log(poses);
```

This produces the output:
```
[
  // pose 1
  {
    // pose score
    "score": 0.42985695206067,
    "keypoints": [
      {
        "position": {
          "x": 126.09371757507,
          "y": 97.861720561981
        },
        "part": "nose",
        "score": 0.99710708856583
      },
      {
        "position": {
          "x": 132.53466176987,
          "y": 86.429876804352
        },
        "part": "leftEye",
        "score": 0.99919074773788
      },
      {
        "position": {
          "x": 100.85626316071,
          "y": 84.421931743622
        },
        "part": "rightEye",
        "score": 0.99851280450821
      },

      ...

      {
        "position": {
          "x": 72.665352582932,
          "y": 493.34189963341
        },
        "part": "rightAnkle",
        "score": 0.0028593824245036
      }
    ],
  },
  // pose 2
  {

    // pose score
    "score": 0.13461434583673,
    "keypoints": [
      {
        "position": {
          "x": 116.58444058895,
          "y": 99.772533416748
        },
        "part": "nose",
        "score": 0.0028593824245036
      }
      {
        "position": {
          "x": 133.49897611141,
          "y": 79.644590377808
        },
        "part": "leftEye",
        "score": 0.99919074773788
      },
      {
        "position": {
          "x": 100.85626316071,
          "y": 84.421931743622
        },
        "part": "rightEye",
        "score": 0.99851280450821
      },

      ...

      {
        "position": {
          "x": 72.665352582932,
          "y": 493.34189963341
        },
        "part": "rightAnkle",
        "score": 0.0028593824245036
      }
    ],
  },
  // pose 2
  {
    // pose score
    "score": 0.13461434583673,
    "keypoints": [
      {
        "position": {
          "x": 116.58444058895,
          "y": 99.772533416748
        },
        "part": "nose",
        "score": 0.0028593824245036
      }
      {
        "position": {
          "x": 133.49897611141,
          "y": 79.644590377808
        },
        "part": "leftEye",
        "score": 0.99919074773788
      },

      ...

      {
        "position": {
          "x": 59.334579706192,
          "y": 485.5936152935
        },
        "part": "rightAnkle",
        "score": 0.004110524430871
      }
    ]
  }
]
```

## Developing the Demos

Details for how to run the demos are included in the `demos/` folder.
