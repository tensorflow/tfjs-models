# Depth Estimation

This package provides models for running depth estimation in TensorFlow.js.

Currently, we provide 1 model option:

## AR Portrait Depth API

This AR portrait depth model estimates per-pixel depth (the distance to the camera center) for a single portrait image, which can be further used for creative applications.
(See [DepthLab](https://augmentedperception.github.io/depthlab/) for potential
applications).

For example, the following demo transforms a single 2D RGB image into a 3D Portrait:
[3D Photo Demo](https://storage.googleapis.com/tfjs-models/demos/3dphoto/index.html)

-------------------------------------------------------------------------------

## Table of Contents

- [How to Run It](#how-to-run-it)
- [Example Code and Demos](#example-code-and-demos)

-------------------------------------------------------------------------------

## How to Run It

There are two steps to run the AR portrait depth API:

First, you create an estimator by choosing one of the models from
`SupportedModels`.

For example:

```javascript
const model = depthEstimation.SupportedModels.ARPortraitDepth;
const estimator = await depthEstimation.createEstimator(model);
```

Next, you can use the estimator to estimate depth.

```javascript
const estimationConfig = {
  minDepth: 0,
  maxDepth: 1,
}
const depthMap = await estimator.estimateDepth(image, estimationConfig);
```

The returned depth map contains depth values for each pixel in the image.

Example output:

```javascript
{
  toCanvasImageSource(): ...
  toArray(): ...
  toTensor(): ...
  getUnderlyingType(): ...
}
```

The output provides access to the underlying depth values using the conversion
functions toCanvasImageSource, toArray, and toTensor depending on the desired
output type. Note that getUnderlyingType can be queried to determine what is the
type being used underneath the hood to avoid expensive conversions (such as from
tensor to image data).

Refer to each model's documentation for specific configurations for the model
and their performance.

[ARPortraitDepth Documentation](https://github.com/tensorflow/tfjs-models/tree/master/depth-estimation/src/ar_portrait_depth)

-------------------------------------------------------------------------------

## Example Code and Demos

You may reference the demos for code examples.
Details for how to run the demos are included in the `demos/`
[folder](https://github.com/tensorflow/tfjs-models/tree/master/depth-estimation/demos).
