# model-name demo

## Setup

Change into the `demo` folder:

```sh
cd model-name/demo
```

Install dependencies:

```sh
yarn
```

Launch a development server, and watch files for changes.

```sh
yarn watch
```

## If you are developing the model locally and want to test the changes in the demo

Change into the `model-name` folder:

```sh
cd model-name
```

Install dependencies:
```sh
yarn
```

Publish `model-name` locally:
```sh
yarn publish-local
```

Change into the demo directory (`model-name/demo`) and install dependencies:

```sh
cd demo
yarn
```

Link the package published from the publish step above:
```sh
yarn link-local
```

Start the dev demo server:
```sh
yarn watch
```

To get future updates from the `model-name` source code, run `yarn publish-local` in the `model-name`
folder again.
