# PoseNet model

This package contains a standalone PoseNet for Pose Estimation,
as well as some demos.

[Refer to the blog post](https://medium.com/p/1cf363e812ce) for a high-level description
of PoseNet in Tensorflow.js.

## Installation

You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/tfjs-posenet"></script>
```

Or you can install it via npm for use in a TypeScript / ES6 project.

```sh
npm install tfjs-posenet --save-dev
```

## Setup

### Developing the Demos

    yarn install

Cd into the demos folder

    cd demos

Install dependencies and prepare the build directory:

    yarn install

To watch files for changes, and launch a dev server:

    yarn watch

## Usage

Either a single pose our multiple poses can be estimated from an image.
Each methodology has its own algorithm and set of parameters.

### Keypoints

All keypoints are indexed by part id.  The parts and their ids are:

| Id | Part |
| -- | -- |
| 0 | nose |
| 1 | left_eye |
| 2 | right_eye |
| 3 | left_ear |
| 4 | right_ear |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_wrist |
| 10 | right_wrist |
| 11 | left_hip |
| 12 | right_hip |
| 13 | left_knee |
| 14 | right_knee |
| 15 | left_ankle |
| 16 | right_ankle |

### Single Pose Estimation

Single pose estimation is the simpler and faster of the two algorithms. Its ideal use case is for when there is only one person in the picture. The downside is that if there is another person in the picture with a keypoint that has a high probability of being accurate, then that keypoint can be associated with the main person’s pose, creating the effect of the two poses being joined together.

```javascript
const pose = poseNet.estimateSinglePose(image, outputStride);
```

#### Inputs

* **image** - a 3d tensor of the image data to predict the poses for.
* **outputStride** - the desired stride for the outputs when feeding the image through the model.  Must be 32, 16, 8.   The higher the number, the faster the performance but slower the accuracy, and visa versa.

#### Returns

It returns a `pose` with a confidence score and an array of keypoints indexed by part id, each with a score and position.

#### Example Usage

##### Estimating a single pose from an image

```javascript
import * as tf from '@tensorflow/tfjs-core';
import {PoseNet} from '@tensorflow/tfjs-models/posenet';
const imageSize = 513;
const outputStride = 16;

async function estimatePoseOnImage(imageElement) {
  const poseNet = new PoseNet();
  await poseNet.load();

  // convert html image element to 3d Tensor
  const image = tf.fromPixels(imageElement);
  // resize image to have acceptable size
  const resized = image.resizeBilinear([imageSize, imageSize]);

  const pose = await poseNet.estimateSinglePose(resized, outputStride);

  image.dispoe();
  resized.dispose();

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
      "point": {
        "x": 301.42237830162,
        "y": 177.69162777066
      },
      "score": 0.99799561500549
    },
    {
      "point": {
        "x": 326.05302262306,
        "y": 122.9596464932
      },
      "score": 0.99766051769257
    },
    {
      "score": 0.99926537275314,
      "point": {
        "x": 258.72196650505,
        "y": 127.51624706388
      }
    },
    {
      "point": {
        "x": 371.96474182606,
        "y": 138.90043857694
      },
      "score": 0.40378707647324
    },
    {
      "point": {
        "x": 203.78961634636,
        "y": 156.80045631528
      },
      "score": 0.78946894407272
    },
    {
      "point": {
        "x": 441.67818045616,
        "y": 338.34006336331
      },
      "score": 0.39678099751472
    },
    {
      "point": {
        "x": 138.2363049984,
        "y": 326.64904239774
      },
      "score": 0.72325360774994
    },
    {
      "point": {
        "x": 486.23655533791,
        "y": 623.74580791593
      },
      "score": 0.014773745089769
    },
    {
      "point": {
        "x": 44.407468557358,
        "y": 521.24995449185
      },
      "score": 0.094393648207188
    },
    {
      "point": {
        "x": 461.8336493969,
        "y": 707.50588062406
      },
      "score": 0.0051866522990167
    },
    {
      "point": {
        "x": 9.5200374126434,
        "y": 541.85546138883
      },
      "score": 0.05102001875639
    },
    {
      "point": {
        "x": 377.21912312508,
        "y": 793.79121902585
      },
      "score": 0.011524646542966
    },
    {
      "point": {
        "x": 146.87340044975,
        "y": 720.60785129666
      },
      "score": 0.0097815711051226
    },
    {
      "point": {
        "x": 359.16951823235,
        "y": 977.52648285031
      },
      "score": 0.0014817604096606
    },
    {
      "point": {
        "x": 135.16723370552,
        "y": 820.65058174729
      },
      "score": 0.0015287395799533
    },
    {
      "point": {
        "x": 347.4669687748,
        "y": 1128.0613844693
      },
      "score": 0.0011711831903085
    },
    {
      "point": {
        "x": 128.20254254341,
        "y": 900.62542334199
      },
      "score": 0.0040716053918004
    }
  ]
}
```

### Multiple Pose Estimation

Multiple Pose estimation can decode multiple poses in a picture. It is more complex and slightly slower than the single pose-algorithm, but has the advantage that if multiple people appear in a picture, their detected keypoints are less likely to be associated with the wrong pose. Even if the use case is to detect a single person’s pose, this algorithm may be more desirable in that the accidental effect of two poses being joined together won’t occur when multiple people appear in the picture. It uses the `Fast greedy decoding` algorithm from the research paper [PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model](https://arxiv.org/pdf/1803.08225.pdf).

```javascript
const poses = await poseNet.estimateMultiplePoses(image, outputStride, maxPoseDetections, scoreThreshold, nmsRadius);
```

#### Inputs

* **image** - a 3d tensor of the image data to predict the poses for.
* **outputStride** - the desired stride for the outputs when feeding the image through the model.  Must be 32, 16, 8.   The higher the number, the faster the performance but slower the accuracy, and visa versa.
* **maxPoseDetections** (optional) - the maximum number of poses to detect. Defaults to 5.
* **scoreThreshold** (optional) - Only return instance detections that have root part score greater or equal to this value. Defaults to 0.5.
* **nmsRadius** (optional) - Non-maximum suppression part distance. It needs to be strictly positive. Two parts suppress each other if they are less than `nmsRadius` pixels away. Defaults to 20.

#### Returns

It returns a `promise` that resolves with an array of `poses`, each with a confidence score and an array of `keypoints` indexed by part id, each with a score and position.

#### Example Usage

##### Estimating multiple poses from an image

```javascript
import * as tf from '@tensorflow/tfjs-core';
import {PoseNet} from '@tensorflow/tfjs-models/posenet';

const imageSize = 513;
const outputStride = 16;
const maxPoseDetections = 2;

const poseNet = new PoseNet();
poseNet.load().then(function() {
  const imageElement = document.getElementById('cat');
  // convert html image element to 3d Tensor
  const image = tf.fromPixels(imageElement);
  // resize image to have acceptable size
  const resized = image.resizeBilinear([imageSize, imageSize]);
  // estimate poses
  const pose = poseNet.estimateMultiplePoses(
    resized, outputStride, maxPoseDetections);

  console.log(poses);

  image.dispoe();
  resized.dispose();
}
```

This produces the output:

```json
[
  {
    "score": 0.42985695206067,
    "keypoints": [
      {
        "point": {
          "x": 126.09371757507,
          "y": 97.861720561981
        },
        "score": 0.99710708856583
      },
      {
        "score": 0.99919074773788,
        "point": {
          "x": 132.53466176987,
          "y": 86.429876804352
        }
      },
      {
        "point": {
          "x": 100.85626316071,
          "y": 84.421931743622
        },
        "score": 0.99851280450821
      },

      ...

      {
        "point": {
          "x": 72.665352582932,
          "y": 493.34189963341
        },
        "score": 0.0028593824245036
      }
    ],
  },
  {
    "score": 0.13461434583673,
    "keypoints": [
      {
        "point": {
          "x": 116.58444058895,
          "y": 99.772533416748
        },
        "score": 0.9978438615799
      },
      {
        "point": {
          "x": 133.49897611141,
          "y": 79.644590377808
        },
        "score": 0.99919074773788
      },

      ...

      {
        "point": {
          "x": 59.334579706192,
          "y": 485.5936152935
        },
        "score": 0.004110524430871
      }
    ]
  }
]
```

##### Getting a keypoint for a specific part in the pose

```javascript
const pose = poseNet.estimateSinglePose(image, outputStride);

const noseKeypoint = pose[jointIds.nose];

const leftKneeKeypoint = post[joinIds.left_knee];

const noseScore = noseKeypoint.score;
const nosePoint = noseKeypoint.point;

const leftKneeScore = leftKneeKeypoint.score;
const leftKneePoint = leftKneeKeypoint.point;
```
