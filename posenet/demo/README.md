# PoseNet Demos

## Contents

### Demo 1: Camera

The camera demo shows how to estimate poses in real-time from a webcam video stream.

<img src="https://raw.githubusercontent.com/tensorflow/tfjs-models/master/posenet/demos/camera.gif" alt="cameraDemo" style="width: 600px;"/>


### Demo 2: Coco Images

The [coco images](http://cocodataset.org/#home) demo shows how to estimate poses in images. It also illustrates the differences between the single-person and multi-person pose detection algorithms.

<img src="https://raw.githubusercontent.com/tensorflow/tfjs-models/master/posenet/demos/coco.gif" alt="cameraDemo" style="width: 600px;"/>


## Setup

cd into the demo folder:

```sh
cd posenet/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing posenet locally, and want to test the changes in the demos

Cd into the posenet folder:
```sh
cd posenet
```

Install dependencies:
```sh
yarn
```

Publish posenet locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
```

Link the local posenet to the demo:
```sh
yarn yalc link @tensorflow-models/posenet
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the posenet source code:
```
# cd up into the posenet directory
cd ../
yarn build && yarn yalc push
```
