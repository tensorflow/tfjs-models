# Hand Detection
MediaPipe Handpose is a lightweight ML pipeline consisting of two models: A palm detector and a hand-skeleton finger tracking model. It predicts 21 3D hand keypoints per detected hand. For more details, please read our Google AI [blogpost](https://ai.googleblog.com/2019/08/on-device-real-time-hand-tracking-with.html).

<img src="demo/demo.gif" alt="demo" style="width:640px" />

Given an input, the model predicts whether it contains a hand. If so, the model returns coordinates for the bounding box around the hand, as well as 21 keypoints within the hand, outlining the location of each finger joint and the palm.

More background information about the model, as well as its performance characteristics on different datasets, can be found here: [https://drive.google.com/file/d/1sv4sSb9BSNVZhLzxXJ0jBv9DqD-4jnAz/view](https://drive.google.com/file/d/1sv4sSb9BSNVZhLzxXJ0jBv9DqD-4jnAz/view)

Check out our [demo](https://storage.googleapis.com/tfjs-models/demos/handtrack/index.html), which uses the model to detect hand landmarks in a live video stream.

This model is also available as part of [MediaPipe](https://hand.mediapipe.dev/), a framework for building multimodal applied ML pipelines.

There are currently two implementations available, a MediaPipe one as well as
a TFJS one. Refer to each model's documentation for specific configurations for the model.

[MPHands TFJS Documentation](https://github.com/tensorflow/tfjs-models/tree/master/handpose/src/tfjs)

[MPHands MediaPipe Documentation](https://github.com/tensorflow/tfjs-models/tree/master/handpose/src/mediapipe)

# Performance

MediaPipe Handpose consists of ~12MB of weights, and is well-suited for real time inference across a variety of devices (40 FPS on a 2018 MacBook Pro, 35 FPS on an iPhone11, 6 FPS on a Pixel3).

# Keypoint Diagram
See the diagram below for the keypoints returned and their their indices.

![MPHands Keypoints](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

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

