# Face landmarks detection demo

## Contents

The face landmarks detection demo shows how to estimate keypoints on a face.

## Setup

cd into the demo folder:

```sh
cd face-landmarks-detection/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing face-landmarks-detection locally, and want to test the changes in the demo

Cd into the face-landmarks-detection folder:
```sh
cd face-landmarks-detection
```

Install dependencies:
```sh
yarn
```

Publish face-landmarks-detection locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
```

Link the local face-landmarks-detection to the demo:
```sh
yarn yalc link @tensorflow-models/face-landmarks-detection
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the face-landmarks-detection source code:
```
# cd up into the face-landmarks-detection directory
cd ../
yarn build && yarn yalc push
```
