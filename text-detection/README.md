# TensorFlow.js Model Starter

A starter pack for implementing mobile-first machine learning models.

## Installation

```bash
# git clone https://github.com/tensorflow/tfjs-models
# cd tfjs-models

export MODEL=model
git clone https://github.com/sdll/tfjs-model-starter.git $MODEL
cd $MODEL
rm -r .git*
```

## Checklist

- [ ] Add a script to convert the model weights serialized in Python to the format [supported](https://github.com/tensorflow/tfjs-converter) by TensorFlow.js to `scripts` (see [examples](./scripts/))
- [ ] Customise `rollup.config.js` and `package.json`, replacing `model-name` with the name of your model and adding a short description
- [ ] Give a poignant name to the main [model class](./src/index.ts) and adjust the [tests](./src/model_test.ts) accordingly
- [ ] Make a beautiful demo of the main functionality (see the [template](./demo/src/index.html)) and showcase the results in the [main](./README.md) and [demo](./demo/README.md) docs

## Setup

- Bundling: [Rollup](./rollup.config.js)
- Formatting: [Prettier](./prettier.config.js)
- Tests: [Jasmine](./run_tests.ts), backed up by the `tfjs-core` utils

## Examples

- [DeepLab](https://github.com/sdll/tfjs-models/tree/models/deeplab/deeplab)
- [EfficientNet](https://github.com/sdll/tfjs-models/tree/models/EfficientNet/efficientnet)