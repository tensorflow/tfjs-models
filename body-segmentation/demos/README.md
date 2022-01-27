# Demos

Try our demos and get inspired with what you can do with body-segmentation models!

## Table of Contents
1. [Live Camera Demo](#live-camera-demo)

2. [Upload a Video Demo](#upload-a-video-demo)

3. [How to Run a Demo](#how-to-run-a-demo)

-------------------------------------------------------------------------------

## Live Camera Demo
This demo uses your camera to get live stream and segments the detected people in real-time.
You can try out different runtimes to see the difference. It
works on laptops, iPhones and android phones.

[MediaPipe SelfieSegmentation model entry](https://storage.googleapis.com/tfjs-models/demos/body-segmentation/index.html?model=selfie_segmentation)

[BodyPix model entry](https://storage.googleapis.com/tfjs-models/demos/body-segmentation/index.html?model=body_pix)

## Upload a Video Demo
This demo allows you to upload a video (in .mp4 format) to run with the model.
Once the video is processed, it automatically downloads the video with segmentation.

[MediaPipe SelfieSegmentation model entry](https://storage.googleapis.com/tfjs-models/demos/body-segmentation-upload-video/index.html?model=selfie_segmentation)

[BodyPix model entry](https://storage.googleapis.com/tfjs-models/demos/body-segmentation-upload-video/index.html?model=body_pix)

## How to Run a Demo
If you want to run any of the demos locally, follow these steps:

1. Go to the demo folder, e.g. `cd live_video`

2. Remove cache etc. `rm -rf .cache dist node_modules`

3. Build dependency. `yarn build-dep`

4. Install dependencies. `yarn`

5. To ensure GPU sync for a correct mediapipe FPS, edit
`./node_modules/@mediapipe/selfie_segmentation/selfie_segmentation.js`
as well as `./node_modules/@mediapipe/pose/pose.js`
and after the statement `y=d.l.getContext("webgl2",{});` add the statement
`window.exposedContext=y;`. This will give the demo access to this context. If
you skip the step then the demo will work but the FPS for mediapipe will be
incorrect.

6. Run the demo. `yarn watch`

7. The demo runs at `localhost:1234`. (Remember to provide URL model parameter e. g. `localhost:1234/?model=mediapipe_hands`)
