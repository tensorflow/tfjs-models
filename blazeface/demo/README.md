# Blazeface demo

## Contents

This demo shows how to use the Blazeface model to detect faces in a video stream.

## Setup

cd into the demos folder:

```sh
cd blazeface/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing blazeface locally, and want to test the changes in the demos

Cd into the blazeface folder:
```sh
cd blazeface
```

Install dependencies:
```sh
yarn
```

Publish blazeface locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local blazeface to the demos:
```sh
yarn yalc link @tensorflow-models/blazeface
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the blazeface source code:
```
# cd up into the blazeface directory
cd ../
yarn build && yarn yalc push
```
