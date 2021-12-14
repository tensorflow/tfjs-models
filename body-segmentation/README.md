# Body Segmentation

This package provides models for running real-time body segmentation.

Currently, we provide 1 model option:

#### MediaPipe SelfieSegmentation:

MediaPipe SelfieSegmentation segments the prominent humans in the scene. It can run in real-time on both smartphones and laptops. The intended use cases include selfie effects and video conferencing, where the person is close (< 2m) to the camera..

-------------------------------------------------------------------------------
## Table of Contents
1. [How to Run It](#how-to-run-it)
2. [Example Code and Demos](#example-code-and-demos)

-------------------------------------------------------------------------------
## How to Run It
In general there are two steps:

You first create a detector by choosing one of the models from `SupportedModels`,
including `MediaPipeSelfieSegmentation`.

For example:

```javascript
const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
const segmenter = await bodySegmentation.createDetector(model);
```

Then you can use the segmenter to segment people in the image.

```
const people = await segmenter.segmentPeople(image);
```

The returned segmentation list contains the detected people in the image.
Note that it is not necessarily the case that there will be one segmentation per
one person. Each model will have its own semantics for the segmentation output.

MediaPipe SelfieSegmentation returns exactly one segmentation corresponding to all people in the input image.

Example output:
```
[
  {
    maskValueToLabel: (maskValue: number) => { return 'person' },
    mask: {
      toCanvasImageSource(): ...
      toImageData(): ...
      toTensor(): ...
      getUnderlyingType(): ...
    }
  }
]
```

The `mask` key stores an object which provides access to the underlying mask image using the conversion functions toCanvasImageSource, toImageData, and toTensor depending on the desired output type. Note that getUnderlyingType can be queried to determine what is the type being used underneath the hood to avoid expensive conversions (such as from tensor to image data).

The semantics of the RGBA values of the `mask` is as follows: the image mask is the same size as the input image, where green and blue channels are always set to 0. Different red values denote different body parts (see maskValueToLabel key below). Different alpha values denote the probability of pixel being a body part pixel (0 being lowest probability and 255 being highest).

`maskValueToLabel` maps a foreground pixel’s red value to the segmented part name of that pixel. Should throw error for unsupported input values. This is not necessarily the same across different models (for example MediaPipeSelfieSegmentation will always return 'person' since it does not distinguish individual body parts).

Refer to each model's documentation for specific configurations for the model
and their performance.

[MediaPipeSelfieSegmentation MediaPipe Documentation](https://github.com/tensorflow/tfjs-models/tree/master/body-segmentation/src/selfie_segmentation_mediapipe)

[MediaPipeSelfieSegmentation TFJS Documentation](https://github.com/tensorflow/tfjs-models/tree/master/body-segmentation/src/selfie_segmentation_tfjs)

-------------------------------------------------------------------------------

## Example Code and Demos
You may reference the demos for code examples. Details for how to run the demos
are included in the `demos/`
[folder](https://github.com/tensorflow/tfjs-models/tree/master/body-segmentation/demos).
