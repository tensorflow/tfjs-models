# BodyPix

Body Segmentation - Body Pix wraps the BodyPix JS Solution within the familiar
TFJS API [BodyPix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix).

This model can be used to segment an image into pixels that are and are not part of a person, and into
pixels that belong to each of twenty-four body parts.  It works for multiple people in an input image or video.

--------------------------------------------------------------------------------

## Table of Contents

1.  [Installation](#installation)
2.  [Usage](#usage)

## Installation

To use BodyPix:

Via script tags:

```html
<!-- Require the peer dependencies of body-segmentation. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter"></script>

<!-- You must explicitly require a TF.js backend if you're not using the TF.js union bundle. -->
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/body-segmentation"></script>
```

Via npm:
```sh
yarn add @tensorflow-models/body-segmentation
yarn add @tensorflow/tfjs-core, @tensorflow/tfjs-converter
yarn add @tensorflow/tfjs-backend-webgl
```

-----------------------------------------------------------------------
## Usage

If you are using the Body Segmentation API via npm, you need to import the libraries first.

### Import the libraries

```javascript
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-converter';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
```

### Create a detector

Pass in `bodySegmentation.SupportedModels.BodyPix` from the
`bodySegmentation.SupportedModel` enum list along with an optional `segmenterConfig` to the
`createSegmenter` method to load and initialize the model.

**By default**, BodyPix loads a MobileNetV1 architecture with a **`0.75`** multiplier.  This is recommended for computers with mid-range/lower-end GPUs.  A model with a **`0.50`** multiplier is recommended for mobile. The ResNet architecture is recommended for computers with even more powerful GPUs.

`segmenterConfig` is an object that defines BodyPix specific configurations for `BodyPixModelConfig`:

 * **architecture** - Can be either `MobileNetV1` or `ResNet50`. It determines which BodyPix architecture to load.

 * **outputStride** - Can be one of `8`, `16`, `32` (Stride `16`, `32` are supported for the ResNet architecture and stride `8`, and `16` are supported for the MobileNetV1 architecture). It specifies the output stride of the BodyPix model. The smaller the value, the larger the output resolution, and more accurate the model at the cost of speed.  ***A larger value results in a smaller model and faster prediction time but lower accuracy***.

 * **multiplier** - Can be one of `1.0`, `0.75`, or `0.50` (The value is used *only* by the MobileNetV1 architecture and not by the ResNet architecture). It is the float multiplier for the depth (number of channels) for all convolution ops. The larger the value, the larger the size of the layers, and more accurate the model at the cost of speed. ***A smaller value results in a smaller model and faster prediction time but lower accuracy***.

 * **quantBytes** - This argument controls the bytes used for weight quantization. The available options are:

   - `4`. 4 bytes per float (no quantization). Leads to highest accuracy and original model size.
   - `2`. 2 bytes per float. Leads to slightly lower accuracy and 2x model size reduction.
   - `1`. 1 byte per float. Leads to lower accuracy and 4x model size reduction.

   The following table contains the corresponding BodyPix 2.0 model checkpoint sizes (widthout gzip) when using different quantization bytes:

     | Architecture       | quantBytes=4 | quantBytes=2 | quantBytes=1 |
     | ------------------ |:------------:|:------------:|:------------:|
     | ResNet50           | ~90MB        | ~45MB        | ~22MB        |
     | MobileNetV1 (1.00) | ~13MB        | ~6MB         | ~3MB         |
     | MobileNetV1 (0.75) | ~5MB         | ~2MB         | ~1MB         |
     | MobileNetV1 (0.50) | ~2MB         | ~1MB         | ~0.6MB       |


* **modelUrl** - An optional string that specifies custom url of the model. This is useful for local development or countries that don't have access to the models hosted on GCP.

```javascript
const model = bodySegmentation.SupportedModels.BodyPix;
const segmenterConfig = {
  architecture: 'ResNet50',
  outputStride: 32,
  quantBytes: 2
};
segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);
```

### Run inference

Now you can use the segmenter to segment people. The `segmentPeople` method
accepts both image and video in many formats, including:
`HTMLVideoElement`, `HTMLImageElement`, `HTMLCanvasElement`, `ImageData`, `Tensor3D`. If you want more
options, you can pass in a second `segmentationConfig` parameter.

`segmentationConfig` is an object that defines BodyPix specific configurations for `BodyPixSegmentationConfig`:

  * **multiSegmentation** - Required. If set to true, then each person is segmented in a separate output, otherwise all people are segmented together in one segmentation.
  * **segmentBodyParts** - Required. If set to true, then 24 body parts are segmented in the output, otherwise only foreground / background binary segmentation is performed.
  * **flipHorizontal** - Defaults to false.  If the segmentation & pose should be flipped/mirrored horizontally.  This should be set to true for videos where the video is by default flipped horizontally (i.e. a webcam), and you want the segmentation & pose to be returned in the proper orientation.
  * **internalResolution** - Defaults to `medium`. The internal resolution percentage that the input is resized to before inference. The larger the `internalResolution` the more accurate the model at the cost of slower prediction times. Available values are `low`, `medium`, `high`, `full`, or a percentage value between 0 and 1. The values `low`, `medium`, `high`, and
`full` map to 0.25, 0.5, 0.75, and 1.0 correspondingly.
  * **segmentationThreshold** - Defaults to 0.7. Must be between 0 and 1. For each pixel, the model estimates a score between 0 and 1 that indicates how confident it is that part of a person is displayed in that pixel.  This *segmentationThreshold* is used to convert these values
to binary 0 or 1s by determining the minimum value a pixel's score must have to be considered part of a person.  In essence, a higher value will create a tighter crop
around a person but may result in some pixels being that are part of a person being excluded from the returned segmentation mask.
  * **maxDetections** -  Defaults to 10. For pose estimation, the maximum number of person poses to detect per image.
  * **scoreThreshold** - Defaults to 0.3. For pose estimation, only return individual person detections that have root part score greater or equal to this value.
  * **nmsRadius** - Defaults to 20. For pose estimation, the non-maximum suppression part distance in pixels. It needs to be strictly positive. Two parts suppress each other if they are less than `nmsRadius` pixels away.

If **multiSegmentation** is set to true then the following additional parameters can be adjusted:

  * **minKeypointScore** - Default to 0.3. Keypoints above the score are used for matching and assigning segmentation mask to each person..
  * **refineSteps** - Default to 10. The number of refinement steps used when assigning the individual person segmentations. It needs to be strictly positive. The larger the higher the accuracy and slower the inference.

The following code snippet demonstrates how to run the model inference:

```javascript
const segmentationConfig = {multiSegmentation: true, segmentBodyParts: false};
const people = await segmenter.segmentPeople(image, segmentationConfig);
```

When `multiSegmentation` is set to false, the returned `people` array contains a single element where all the people segmented in the image are found in that single segmentation element. When `multiSegmentation` is set to true, then the length of the array will be equal to the number of detected people, each segmentation containing one person.

When `segmentBodyParts` is set to false, the only label returned by the maskValueToLabel function is 'person'. When `segmentBodyParts` is set to true, the maskValueToLabel function will return one of the body parts defined by BodyPix, where the mapping of mask values to label is as follows:

| Part Id | Part Name              | Part Id | Part Name              |
|---------|------------------------|---------|------------------------|
| 0       | left_face              | 12      | torso_front            |
| 1       | right_face             | 13      | torso_back             |
| 2       | left_upper_arm_front   | 14      | left_upper_leg_front   |
| 3       | left_upper_arm_back    | 15      | left_upper_leg_back
| 4       | right_upper_arm_front  | 16      | right_upper_leg_front
| 5       | right_upper_arm_back   | 17      | right_upper_leg_back
| 6       | left_lower_arm_front   | 18      | left_lower_leg_front
| 7       | left_lower_arm_back    |  19      | left_lower_leg_back
| 8       | right_lower_arm_front  | 20      | right_lower_leg_front
| 9       | right_lower_arm_back   | 21      | right_lower_leg_back
| 10      | left_hand              | 22      | left_foot
| 11      | right_hand             | 23      | right_foot


Please refer to the Body Segmentation API
[README](https://github.com/tensorflow/tfjs-models/blob/master/body-segmentation/README.md#how-to-run-it)
about the structure of the returned `people` array.
