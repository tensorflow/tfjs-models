# Handtrack demo

## Contents

This demo shows how to use the handtrack model to detect faces in a video stream.

## Setup

cd into the demos folder:

```sh
cd handtrack/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing handtrack locally, and want to test the changes in the demos

Cd into the handtrack folder:
```sh
cd handtrack
```

Install dependencies:
```sh
yarn
```

Publish handtrack locally:
```sh
yarn build && yarn yalc publish
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local handtrack to the demos:
```sh
yarn yalc link @tensorflow-models/handtrack
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the handtrack source code:
```
# cd up into the handtrack directory
cd ../
yarn build && yarn yalc push
```
