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

Launch a development server, and watch files for changes. This command will also automatically open
the demo app in your browser.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

`cd` into the coco-ssd/ folder:

```sh
cd coco-ssd
```

Install dependencies:
```sh
yarn
```

Publish coco-ssd locally:
```sh
yarn publish-local
```

`cd` into this directory (coco-ssd/demo) and install dependencies:

```sh
cd demo
yarn
```

Link the package published from the publish step above:
```sh
yarn link-local
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `coco-ssd` source code, just run `yarn publish-local` in the coco-ssd/
folder again.
