# Toxicity classifier demo

## Contents

The demo shows how to use predictions produced by the Toxicity classifier.

## Setup

cd into the demo folder:

```sh
cd toxicity/demo
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To watch files for changes, and launch a dev server:

```sh
yarn watch
```

## If you are developing toxicity locally, and want to test the changes in the demo

cd into the toxicity folder:
```sh
cd toxicity
```

Install dependencies:
```sh
yarn
```

Build and publish toxicity locally:
```sh
yarn publish-local
```

Cd into the demo and install dependencies:

```sh
cd demo
yarn
```

Link the local toxicity to the demo:
```sh
yarn link-local
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the toxicity source code:
```
# cd up into the toxicity directory
cd ../
yarn publish-local
```
