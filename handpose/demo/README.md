# Handpose demo

## Contents

This demo shows how to use the handpose model to detect hands in a video stream.

## Setup

cd into the demos folder:

```sh
cd handpose/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing handpose locally, and want to test the changes in the demos

Cd into the handpose folder:
```sh
cd handpose
```

Install dependencies:
```sh
yarn
```

Publish handpose locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local handpose to the demos:
```sh
yarn yalc link @tensorflow-models/handpose
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the handpose source code:
```
# cd up into the handpose directory
cd ../
yarn build && yarn yalc push
```
