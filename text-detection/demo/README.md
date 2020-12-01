# Text Detection Demo

## Setup

Change the directory to the `demo` folder:

```sh
cd text-detection/demo
```

Install dependencies:

```sh
yarn
```

Launch the development server watching the files for changes.

```sh
yarn watch
```

## Development

If you are developing the model locally and want to test the changes in the demo, proceed as follows:

### Change the directory to the `text-detection` folder

```sh
cd text-detection
```

### Install dependencies

```sh
yarn
```

### Publish a local copy o text-detectionf

```sh
yarn publish-local
```

### Change into the demo directory (`text-detection/demo`) and install dependencies

```sh
cd demo
yarn
```

### Link the package published from the publish step above

```sh
yarn link-local
```

### Start the dev demo server

```sh
yarn watch
```

**Note**: *To get future updates from the `text-detection` source code, just run `yarn publish-local` in the `text-detection` folder again.*
