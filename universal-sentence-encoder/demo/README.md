# Universal Sentence Encoder Demo

## Contents

The demo shows how to use embeddings produced by the Universal Sentence Encoder.

## Setup

cd into the demo folder:

```sh
cd universal-sentence-encoder/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

Build the universal sentence encoder locally which the demo depends on:

```sh
yarn build-deps
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing universal-sentence-encoder locally, and want to test the changes in the demo

cd into the universal-sentence-encoder/demo folder:
```sh
cd universal-sentence-encoder/demo
```

Rebuild universal sentence encoder locally:
```sh
yarn build-deps
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `universal-sentence-encoder` source code,
just run `yarn build-deps` in the universal-sentence-encoder/demo folder again.
