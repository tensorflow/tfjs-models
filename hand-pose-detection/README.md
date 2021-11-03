# Hand Pose Detection

This package provides models for running real-time hand pose detection.

Currently, we provide 1 model option:

#### MediaPipe:
MediaPipe Hands can detect multiple hands, each hand contains 21 hand keypoints.

-------------------------------------------------------------------------------
## Table of Contents
1. [How to Run It](#how-to-run-it)
2. [Keypoint Diagram](#keypoint-diagram)

-------------------------------------------------------------------------------
## How to Run It
In general there are two steps:

You first create a detector by choosing one of the models from `SupportedModels`,
including `MediaPipeHands`.

For example:

```javascript
const model = handPoseDetection.SupportedModels.MediaPipeHands;
const detector = await handPoseDetection.createDetector(model);
```

Then you can use the detector to detect hands.

```
const hands = await detector.estimateHands(image);
```

The returned hands list contains detected hands for each hand in the image.
If the model cannot detect any hands, the list will be empty.

For each hand, it contains a prediction of the handedness (left or right), a confidence score of this prediction, as well as an array of keypoints.
MediaPipeHands returns 21 keypoints.
Each keypoint contains x, y, z, and name.

Example output:
```
[
  {
    score: 0.8,
    Handedness: ‘Right’,
    keypoints: [
      {x: 105, y: 107, z: -15, name: "wrist"},
      {x: 108, y: 160, z: -40, name: "pinky_tip"},
      ...
    ]
  }
]
```

For the `keypoints`, x and y represent the actual keypoint position in the image.
z represents the landmark depth with the depth at the wrist being the origin, and the smaller the value the closer the landmark is to the camera.

The name provides a label for each keypoint, such as 'wrist', 'pinky_tip', etc.

Refer to each model's documentation for specific configurations for the model
and their performance.

[MediaPipeHands MediaPipe Documentation](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection/src/mediapipe)

[MediaPipeHands TFJS Documentation](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection/src/tfjs)

-------------------------------------------------------------------------------

## Keypoint Diagram
See the diagram below for what those keypoints are and their index in the array.

### MediaPipe Hands Keypoints: Used in MediaPipe Hands
![MediaPipeHands Keypoints](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)
0: wrist  \
1: thumb_cmc \
2: thumb_mcp  \
3: thumb_ip  \
4: thumb_tip  \
5: index_finger_mcp  \
6: index_finger_pip  \
7: index_finger_dip  \
8: index_finger_tip  \
9: middle_finger_mcp  \
10: middle_finger_pip  \
11: middle_finger_dip  \
12: middle_finger_tip  \
13: ring_finger_mcp  \
14: ring_finger_pip  \
15: ring_finger_dip  \
16: ring_finger_tip  \
17: pinky_finger_mcp  \
18: pinky_finger_pip  \
19: pinky_finger_dip  \
20: pinky_finger_tip
