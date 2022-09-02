# Object Detection (COCO-SSD) Demo

This demo allows you to try out object detection on a couple of preset images using different base models.

## Setup

`cd` into the demo/ folder:

```sh
cd coco-ssd/demo
```

Install dependencies:

```sh
yarn
```

Build the coco-ssd model locally which the demo depends on:

```sh
yarn build-deps
```

Launch a development server, and watch files for changes. This command will also automatically open
the demo app in your browser.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

`cd` into the coco-ssd/demo folder:

```sh
cd coco-ssd/demo
```

Rebuild coco-ssd locally:
```sh
yarn build-deps
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `coco-ssd` source code, just run `yarn build-deps` in the coco-ssd/demo
folder again.
