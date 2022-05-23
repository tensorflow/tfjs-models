# Face Detection

This package provides models for running real-time face detection.

Currently, we provide 1 model option:

#### MediaPipe FaceDetection:
[Demo](https://storage.googleapis.com/tfjs-models/demos/face-detection/index.html?model=mediapipe_face_detector)

MediaPipe FaceDetection can detect multiple faces, each face contains 6 keypoints.

More background information about the package, as well as its performance characteristics on different datasets, can be found here: [Short Range Model Card](https://drive.google.com/file/d/1d4-xJP9PVzOvMBDgIjz6NhvpnlG9_i0S/preview), [Sparse Full Range Model Card](https://drive.google.com/file/d/1aZtpSwsBhA1Epd-ZDfwoQYSTQwEfLm5Z/preview).

-------------------------------------------------------------------------------
## Table of Contents
1. [How to Run It](#how-to-run-it)
2. [Example Code and Demos](#example-code-and-demos)

-------------------------------------------------------------------------------
## How to Run It
In general there are two steps:

You first create a detector by choosing one of the models from `SupportedModels`, including `MediaPipeFaceDetector`.

For example:

```javascript
const model = faceDetection.SupportedModels.MediaPipeFaceDetector;
const detectorConfig = {
  runtime: 'mediapipe', // or 'tfjs'
}
const detector = await faceDetection.createDetector(model, detectorConfig);
```

Then you can use the detector to detect faces.

```
const faces = await detector.estimateFaces(image);
```

The returned face list contains detected faces for each face in the image.
If the model cannot detect any faces, the list will be empty.

For each face, it contains a bounding box of the detected face, as well as an array of keypoints. `MediaPipeFaceDetector` returns 6 keypoints.
Each keypoint contains x and y, as well as a name.

Example output:
```
[
  {
    box: {
      xMin: 304.6476503248806,
      xMax: 502.5079975897382,
      yMin: 102.16298762367356,
      yMax: 349.035215984403,
      width: 197.86034726485758,
      height: 246.87222836072945
    },
    keypoints: [
      {x: 446.544237446397, y: 256.8054528661723, name: "rightEye"},
      {x: 406.53152857172876, y: 255.8, "leftEye },
      ...
    ],
  }
]
```

The `box` represents the bounding box of the face in the image pixel space, with `xMin`, `xMax` denoting the x-bounds, `yMin`, `yMax` denoting the y-bounds, and `width`, `height` are the dimensions of the bounding box.

For the `keypoints`, x and y represent the actual keypoint position in the image pixel space.

The name provides a label for the keypoint, which are 'rightEye', 'leftEye', 'noseTip', 'mouthCenter', 'rightEarTragion', and 'leftEarTragion' respectively.

Refer to each model's documentation for specific configurations for the model
and their performance.

[MediaPipeFaceDetector MediaPipe Documentation](https://github.com/tensorflow/tfjs-models/tree/master/face-detection/src/mediapipe)

[MediaPipeFaceDetector TFJS Documentation](https://github.com/tensorflow/tfjs-models/tree/master/face-detection/src/tfjs)

## Example Code and Demos
You may reference the demos for code examples. Details for how to run the demos
are included in the `demos/`
[folder](https://github.com/tensorflow/tfjs-models/tree/master/face-detection/demos).
