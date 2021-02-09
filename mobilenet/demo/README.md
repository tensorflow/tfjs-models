# Mobilenet Demo

This demo allows you to try out image classification using the mobilenet model.

## Setup

`cd` into the demo/ folder:

```sh
cd mobilenet/demo
```

Install dependencies:

```sh
yarn
```

Build the mobilenet model locally which the demo depends on:

```sh
yarn build-deps
```

Launch a development server, and watch files for changes. This command will also automatically open
the demo app in your browser.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

`cd` into the mobilenet/demo folder:

```sh
cd mobilenet/demo
```

Rebuild mobilenet locally:
```sh
yarn build-deps
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `mobilenet` source code, just run `yarn build-deps` in the mobilenet/demo
folder again.
