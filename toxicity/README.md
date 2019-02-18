# Toxicity classifier

This TensorFlow.js module is a multi-head toxicity classifier built on top of the Universal Sentence Encoder ([Cer et al., 2018](https://arxiv.org/pdf/1803.11175.pdf)). The classifier predicts whether the input text should be assigned any of the seven following labels (prediction heads):

`toxicity`
`severe_toxicity`
`identity_attack`
`insult`
`threat`
`sexual_explicit`
`obscene`

More information about how each label was calibrated can be found [here](https://github.com/conversationai/conversationai.github.io/blob/master/crowdsourcing_annotation_schemes/toxicity_with_subattributes.md).

[Try the demo here.](https://storage.googleapis.com/tfjs-models/demos/toxicity/index.html)

![demo](./images/demo.jpg)

In this demo, we predict the toxicity of several sentences taken from this [Kaggle dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). Users can also input their own text for classification.

## Usage

To import in npm:

```js
import * as toxicity from '@tensorflow-models/toxicity';
```

or as a standalone script tag:

```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/toxicity"></script>
```

Then:

```js
// The minimum prediction confidence.
const threshold = 0.9;

// Load the model. Users optionally pass in a threshold and an array of
// prediction heads to include.
toxicity.load(threshold).then(model => {
  const sentences = ['you suck'];

  model.classify(sentences).then(predictions => {
    // `predictions` is an array of objects, one for each prediction head,
    // that contains the raw probabilities for each input along with the
    // final prediction boolean given the threshold.

    console.log(predictions);
    /*
    prints:
    {
      "label": "identity_attack",
      "results": [{
        "probabilities": [0.9659664034843445, 0.03403361141681671],
        "match": false
      }]
    },
    {
      "label": "insult",
      "results": [{
        "probabilities": [0.08124706149101257, 0.9187529683113098],
        "match": true
      }]
    },
    ...
     */
  });
});
```