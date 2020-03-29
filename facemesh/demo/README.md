# MediaPipe Facemesh demo

## Contents

The MediaPipe Facemesh demo shows how to use the MediaPipe Facemesh model to estimate keypoints on a face.

## Setup

cd into the demo folder:

```sh
cd facemesh/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing facemesh locally, and want to test the changes in the demos

Cd into the facemesh folder:
```sh
cd facemesh
```

Install dependencies:
```sh
yarn
```

Publish facemesh locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demos and install dependencies:

```sh
cd demo
yarn
```

Link the local facemesh to the demos:
```sh
yarn yalc link @tensorflow-models/facemesh
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the facemesh source code:
```
# cd up into the facemesh directory
cd ../
yarn build && yarn yalc push
```
