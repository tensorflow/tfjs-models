import * as tfconv from '@tensorflow/tfjs-converter';
import * as semanticSegmentation from '@tensorflow-models/deeplab';
const loadSemanticSegmentation = async () => {
  const base = 'pascal';        // set to your preferred model, out of `pascal`,
                                // `cityscapes` and `ade20k`
  const quantizationBytes = 2;  // either 1, 2 or 4
  // use the getURL utility function to get the URL to the pre-trained weights
  const modelUrl = semanticSegmentation.getURL(base, quantizationBytes);
  const rawModel = await tfconv.loadGraphModel(modelUrl);
  const modelName = 'pascal';  // set to your preferred model, out of `pascal`,
  // `cityscapes` and `ade20k`
  return new semanticSegmentation.SemanticSegmentation(rawModel);
};
loadSemanticSegmentation().then(() => console.log(`Loaded the model successfully!`));
