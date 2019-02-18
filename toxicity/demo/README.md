# Toxicity classifier demo

## Contents

The demo shows how to use predictions produced by the Toxicity classifier.

## Setup

cd into the demos folder:

```sh
cd toxicity/demos
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing toxicity locally, and want to test the changes in the demos

Install yalc:
```sh
npm i -g yalc
```

cd into the toxicity folder:
```sh
cd toxicity
```

Install dependencies:
```sh
yarn
```

Publish toxicity locally:
```sh
yalc push
```

Cd into the demos and install dependencies:

```sh
cd demos
yarn
```

Link the local toxicity to the demos:
```sh
yalc link @tensorflow-models/toxicity
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the toxicity source code:
```
# cd up into the toxicity directory
cd ../
yarn build && yalc push
```
