# EfficientNet Demo

This demo allows you to classify images using EfficientNet and measure its performance.

## Setup

Change the directory to the `demo` folder:

```sh
cd efficientnet/demo
```

Install dependencies:

```sh
yarn
```

Launch the development server watching the files for changes.

```sh
yarn watch
```

**Warning**: *The higher the model version, the more time the inference takes.*

## Development

If you are developing the model locally and want to test the changes in the demo, proceed as follows:

### Change the directory to the `efficientnet` folder

```sh
cd efficientnet
```

### Install dependencies

```sh
yarn
```

### Publish a local copy of efficientnet

```sh
yarn publish-local
```

### Change into the demo directory (`efficientnet/demo`) and install dependencies

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

**Note**: *To get future updates from the `efficientnet` source code, just run `yarn publish-local` in the `efficientnet` folder again.*
