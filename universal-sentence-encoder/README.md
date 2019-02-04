# Universal Sentence Encoder lite

The Universal Sentence Encoder ([Cer et al., 2018](https://arxiv.org/pdf/1803.11175.pdf)) is a model that encodes text into 512-dimensional embeddings. These embeddings can then be used as inputs to natural language processing tasks such as [sentiment classification](https://en.wikipedia.org/wiki/Sentiment_analysis) and [textual similarity](https://en.wikipedia.org/wiki/Semantic_similarity) analysis.

This module is a TensorFlow.js [`FrozenModel`](https://js.tensorflow.org/api/latest/#loadFrozenModel) converted from the Universal Sentence Encoder lite ([module on TFHub](https://tfhub.dev/google/universal-sentence-encoder-lite/2)), a lightweight version of the original. The lite model is based on the Transformer ([Vaswani et al, 2017](https://arxiv.org/pdf/1706.03762.pdf)) architecture, and uses an 8k word piece [vocabulary](https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/vocab.json).

## Usage

```js

import * as use from '@tensorflow-models/universal-sentence-encoder';

// Load the model.
const model = await use.load();

// Embed an array of sentences.
const sentences = [
  'Hello.',
  'How are you?'
];

const embeddings = await model.embed(sentences);

// `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
// So in this example `embeddings` has the shape [2, 512].
const verbose = true;
embeddings.print(verbose);

```