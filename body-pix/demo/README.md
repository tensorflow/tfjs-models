# body-pix Demos

## Contents

The demo shows how to estimate segmentation in real-time from a webcam video stream.

## Setup

cd into the demo folder:

```sh
cd body-pix/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing body-pix locally, and want to test the changes in the demo

Cd into the body-pix folder:
```sh
cd body-pix
```

Install dependencies:
```sh
yarn
```

Build and publish the body-pix locally:
```sh
yarn publish-local
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
```

Link the local body-pix to the demo:
```sh
yarn yalc link @tensorflow-models/body-pix
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the body-pix source code:
```
# cd up into the body-pix directory
cd ../
yarn build && yalc push
```
