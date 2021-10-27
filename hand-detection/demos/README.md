# Demos

Try our demos and get inspired with what you can do with hand-detection models!

## Table of Contents
1. [Live Camera Demo](#live-camera-demo)

2. [How to Run a Demo](#how-to-run-a-demo)

-------------------------------------------------------------------------------

## Live Camera Demo
This demo uses your camera to get live stream and tracks your hands in real-time.
You can try out different runtimes to see the difference. It
works on laptops, iPhones and android phones.

[MediaPipeHands model entry](https://storage.googleapis.com/tfjs-models/demos/hand-detection/index.html?model=mediapipe_hands)

## How to Run a Demo
If you want to run any of the demos locally, follow these steps:

1. Go to the demo folder, e.g. `cd live_video`

2. Remove cache etc. `rm -rf .cache dist node_modules`

3. Build dependency. `yarn build-dep`

4. Install dependencies. `yarn`

5. Run the demo. `yarn watch`

6. The demo runs at `localhost:1234`. (Remember to provide URL model parameter e. g. `localhost:1234/?model=mediapipe_hands`)
