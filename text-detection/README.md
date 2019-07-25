# In-Browser Shape Robust Text Detection with Progressive Scale Expansion Network

## This model is a work-in-progress and has not been released yet. We will update this README when the model is released and usable

![Text Detection Demo](./docs/demo.gif)

This model is a TensorFlow.js implementation of a versatile text detector based on [PSENet](https://github.com/liuheng92/tensorflow_PSENet), which can detect text of arbitrary shape.

Using the model does not require any specific knowledge about machine learning. It can take any browser-based image elements (`<img>`, `<video>` and `<canvas>` elements, for example) as input return an array of bounding boxes.

## Usage

Four parameters affect the accuracy, precision and speed of inference:

* **degree of quantization**

  Three types of weights are supported by default, quantized either to 1, 2 or 4 bytes respectively. The greater the degree of quantization, the less the size and precision of the output (the current model quantized to **1 byte** weighs **29 MB**, to **2 bytes** ⁠— **59 MB**, to **4 bytes** ⁠— **115 MB**)

* **resize length**

  The input image is [resized](./src/utils.ts#L133) before being processed by the model. The greater the length limiting the maximum side of the resized image, the more accurate the results, but the inference takes significantly more time and memory.

* **minimum textbox area**

  Increasing this parameter, you can avoid spurious predictions, improving their accuracy.

* **minimum confidence**

  As an intermediate step, the model generates confidence levels to dermine whether each pixel represents text or not. Setting this threshold to higher values might improve the robustness of predictions.

## Technical Details

PSENet operates in two steps (see the latest paper [here](https://arxiv.org/abs/1903.12473)):

![PSENet pipeline](./docs/pipeline.png)

* First, the model resizes the input and generates a series of two-dimensional *minimal scale kernels* (the current implementation yields 6), all having the same size as the scaled input.

* The Progressive Scale Expansion algorithm (PSE) then processes these kernels, extracting text pixels.

Due to the fact that each minimal scale kernel stores distinctive information about the geometry of the original image, this method is effective for differentiating between close neighbours.
