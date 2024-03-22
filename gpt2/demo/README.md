# GPT2 demo

## Contents

This demo shows how to use the GPT2 model to generate text.

## Setup

cd into the `gpt2` folder. From the root of the repo, this is located at `gpt2/`. From the demo, it's `../`.

Compares the actual contents of node_modules with the expected contents listed in yarn.lock.  If any dependencies are missing, the command installs them.
```sh
yarn build-deps
```

Install dependencies:
```sh
yarn
```

cd into the demo and install dependencies:

Compares the actual contents of node_modules with the expected contents listed in yarn.lock.  If any dependencies are missing, the command installs them.
```sh
yarn build-deps
```

```sh
cd demo
yarn
```

build the demo's dependencies. You'll need to re-run this whenever you make changes to the `@tfjs-models/gpt2` package that this demo uses.
```sh
yarn build-deps
```

start the dev demo server:
```sh
yarn watch
```
