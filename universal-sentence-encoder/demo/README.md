# Universal Sentence Encoder Demo

## Contents

The demo shows how to use embeddings produced by the Universal Sentence Encoder.

## Setup

cd into the demos folder:

```sh
cd universal-sentence-encoder/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing universal-sentence-encoder locally, and want to test the changes in the demos

Install yalc:
```sh
npm i -g yalc
```

cd into the universal-sentence-encoder folder:
```sh
cd universal-sentence-encoder
```

Install dependencies:
```sh
yarn
```

Publish universal-sentence-encoder locally:
```sh
yalc push
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local universal-sentence-encoder to the demos:
```sh
yalc link @tensorflow-models/universal-sentence-encoder
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the universal-sentence-encoder source code:
```
# cd up into the universal-sentence-encoder directory
cd ../
yarn build && yalc push
```
