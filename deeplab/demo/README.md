# DeepLab Demo

This demo allows you to try out semantic segmentation on a couple of preset images using different base models.

## Setup

Run `yarn` in the root and 'deeplab' folder to install dependencies.

Run `yarn build-npm` in the 'deeplab' folder.

Change the directory to the `demo` folder:

```sh
cd deeplab/demo
```

Install dependencies:

```sh
yarn
```

Launch the development server watching the files for changes.

```sh
yarn watch
```

**Warning**: *Running the Cityscapes model in the demo is resource-intensive and might crash your browser.*

## Development

If you are developing the model locally and want to test the changes in the demo, proceed as follows:

### Change the directory to the `deeplab` folder

```sh
cd deeplab
```

### Install dependencies

```sh
yarn
```

### Publish a local copy of deeplab

```sh
yarn publish-local
```

### Change into the demo directory (`deeplab/demo`) and install dependencies

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

**Note**: *To get future updates from the `deeplab` source code, just run `yarn publish-local` in the `deeplab` folder again.*
