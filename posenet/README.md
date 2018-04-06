# PoseNet model

This package contains a standalone PoseNet model for classification,
as well as some demos.

## Setup

### Developing the Demos

Install dependencies and prepare the build directory:

    yarn prep

To watch files for changes, and launch a dev server:

    yarn dev

### To import this as a dependency

Clone this repository:

   git clone https://creativelab-internal.googlesource.com/cl-deeplearnjs-posenet

Build and link it.  This registers the local `posenet` as a global module on your computer

   yarn prep
   yarn build
   yarn link

Then, in **your repository:

   yarn link deeplearn-posenet

## To deploy the demo

   yarn build-demo
   yarn deploy-demo
