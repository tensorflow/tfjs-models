# In-Browser Shape Robust Text Detection with Progressive Scale Expansion Network

This model is a TensorFlow.js implementation of a versatile text detector based on PSENet, which can precisely detect text instances with arbitrary shapes.

Using the model does not require any specific knowledge about machine learning. It can take any browser-based image elements (<img>, <video> and <canvas> elements, for example) as input return an array of bounding boxes.

## Technical Details

PSENet generates the different scale of kernels for each text instance, and gradually expands the minimal scale kernel to the text instance with the complete shape. Due to the fact that there are large geometrical margins among the minimal scale kernels, this method is effective to split the close text instances, making it easier to use segmentation-based methods to detect arbitrary-shaped text instances.
