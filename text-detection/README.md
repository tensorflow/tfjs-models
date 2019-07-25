# In-Browser Shape Robust Text Detection with Progressive Scale Expansion Network

## This model is a work-in-progress and has not been released yet. We will update this README when the model is released and usable

![Text Detection Demo](./docs/demo.gif)

This model is a TensorFlow.js implementation of a versatile text detector based on PSENet, which can detect text of arbitrary shape.

Using the model does not require any specific knowledge about machine learning. It can take any browser-based image elements (`<img>`, `<video>` and `<canvas>` elements, for example) as input return an array of bounding boxes.


## Technical Details

PSENet operates in two steps.

![PSENet pipeline](./docs/pipeline.png)

* First, the model resizes the input and generates a series of two-dimensional *minimal scale kernels* (the current implementation yields 6), all having the same size as the scaled input.

* The Progressive Scale Expansion algorithm (PSE) then processes these kernels, extracting text pixels.

Due to the fact that each minimal scale kernel stores distinctive information about the geometry of the original image, this method is effective for differentiating between close neighbours.
