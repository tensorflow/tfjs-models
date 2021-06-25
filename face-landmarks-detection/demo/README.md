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

Build the model locally which the demo depends on:

```sh
yarn build-deps
```

Launch a development server, and watch files for changes.

```sh
yarn watch
```

## If you are developing face-landmarks-detection locally, and want to test the changes in the demo

Cd into the face-landmarks-detection/demo folder:
```sh
cd face-landmarks-detection/demo
```

Rebuild the model locally:
```sh
yarn build-deps
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `face-landmarks-detection` source code, just run
`yarn build-deps` in the face-landmarks-detection/demo folder again.
