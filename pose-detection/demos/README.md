# Demos

Try our demos and get inspired with what you can do with pose-detection models!

## Table of Contents
1.  [Live Camera Demo](#live-camera-demo)

2. [Upload a Video Demo](#upload-a-video-demo)

3. [How to Run a Demo](#how-to-run-a-demo)

-------------------------------------------------------------------------------

## Live Camera Demo
This demo uses your camera to get live stream and tracks your poses in real-time.
You can try out different models, runtimes and backends to see the difference. It
works on laptops, iPhones and android phones.

[MoveNet model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=movenet)

[BlazePose model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=blazepose)

[PoseNet model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection/index.html?model=posenet)


## Upload a Video Demo
This demo allows you to upload a video (in .mp4 format) to run with the model.
Once the video is processed, it automatically downloads the video with pose keypoints.

[MoveNet model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection-upload-video/index.html?model=movenet)

[BlazePose model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection-upload-video/index.html?model=blazepose)

[PoseNet model entry](https://storage.googleapis.com/tfjs-models/demos/pose-detection-upload-video/index.html?model=posenet)


## How to Run a Demo
If you want to run any of the demos locally, follow these steps:

1. Go to the demo folder, e.g. `cd live_video`

2. Remove cache etc. `rm -rf .cache dist node_modules`

3. Build dependency. `yarn build-dep`

4. Install dependencies. `yarn`

5. Run the demo. `yarn watch`

6. The demo runs at `localhost:1234`. (Remember to provide URL model parameter e. g. `localhost:1234/?model=movenet`)
