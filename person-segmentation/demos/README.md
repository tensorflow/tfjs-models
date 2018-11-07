# person-segmentation Demos

## Contents

The demo shows how to estimate segmentation in real-time from a webcam video stream.

## Setup

cd into the demos folder:

```sh
cd person-segmentation/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing person-segmentation locally, and want to test the changes in the demos

Install yalc:
```sh
npm i -g yalc
```

cd into the person-segmentation folder:
```sh
cd person-segmentation
```

Install dependencies:
```sh
yarn
```

Publish person-segmentation locally:
```sh
yalc push
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local person-segmentation to the demos:
```sh
yalc link \@tensorflow-models/person-segmentation
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the person-segmentation source code:
```
# cd up into the person-segmentation directory
cd ../
yarn build && yalc push
```
