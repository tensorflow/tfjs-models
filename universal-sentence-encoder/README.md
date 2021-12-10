# Universal Sentence Encoder lite

The Universal Sentence Encoder ([Cer et al., 2018](https://arxiv.org/pdf/1803.11175.pdf)) (USE) is a model that encodes text into 512-dimensional embeddings. These embeddings can then be used as inputs to natural language processing tasks such as [sentiment classification](https://en.wikipedia.org/wiki/Sentiment_analysis) and [textual similarity](https://en.wikipedia.org/wiki/Semantic_similarity) analysis.

This module is a TensorFlow.js [`GraphModel`](https://js.tensorflow.org/api/latest/#loadGraphModel) converted from the USE lite ([module on TFHub](https://tfhub.dev/google/universal-sentence-encoder-lite/2)), a lightweight version of the original. The lite model is based on the Transformer ([Vaswani et al, 2017](https://arxiv.org/pdf/1706.03762.pdf)) architecture, and uses an 8k word piece [vocabulary](https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/vocab.json).

In [this demo](./demo/index.js) we embed six sentences with the USE, and render their self-similarity scores in a matrix (redder means more similar):

![selfsimilarity](https://storage.googleapis.com/tfjs-models/assets/use/self_similarity.jpg)

*The matrix shows that USE embeddings can be used to cluster sentences by similarity.*

The sentences (taken from the [TensorFlow Hub USE lite colab](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder_lite.ipynb#scrollTo=_GSCW5QIBKVe)):
1. I like my phone.
2. Your cellphone looks great.
3. How old are you?
4. What is your age?
5. An apple a day, keeps the doctors away.
6. Eating strawberries is healthy.

# Universal Sentence Encoder For Question Answering

The Universal Sentence Encoder for question answering (USE QnA) is a model that encodes question and answer texts into 100-dimensional embeddings. The dot product of these embeddings measures how well the answer fits the question. It can also be used in other applications, including any type of text classification, clustering, etc.
This module is a lightweight TensorFlow.js [`GraphModel`](https://js.tensorflow.org/api/latest/#loadGraphModel). The model is based on the Transformer ([Vaswani et al, 2017](https://arxiv.org/pdf/1706.03762.pdf)) architecture, and uses an 8k SentencePiece [vocabulary](https://tfhub.dev/google/tfjs-model/universal-sentence-encoder-qa-ondevice/1/vocab.json?tfjs-format=file). It is trained on a variety of data sources, with the goal of learning text representations that are useful out-of-the-box to retrieve an answer given a question.

In [this demo](./demo/index.js) we embed a question and three answers with the USE QnA, and render their their scores:

![QnA scores](https://storage.googleapis.com/tfjs-models/assets/use/qna_score.png)

*The scores show how well each answer fits the question.*

## Installation

Using `yarn`:

    $ yarn add @tensorflow/tfjs @tensorflow-models/universal-sentence-encoder

Using `npm`:

    $ npm install @tensorflow/tfjs @tensorflow-models/universal-sentence-encoder

## Usage

To import in npm:

```js
require('@tensorflow/tfjs');
const use = require('@tensorflow-models/universal-sentence-encoder');
```

or as a standalone script tag:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/universal-sentence-encoder"></script>
```

Then:

```js
// Load the model.
use.load().then(model => {
  // Embed an array of sentences.
  const sentences = [
    'Hello.',
    'How are you?'
  ];
  model.embed(sentences).then(embeddings => {
    // `embeddings` is a 2D tensor consisting of the 512-dimensional embeddings for each sentence.
    // So in this example `embeddings` has the shape [2, 512].
    embeddings.print(true /* verbose */);
  });
});
```

`load()` accepts an optional configuration object where you can set custom `modelUrl` and/or `vocabUrl` strings (e.g. `use.load({modelUrl: '', vocabUrl: ''})`).

To use the Tokenizer separately:

```js
use.loadTokenizer().then(tokenizer => {
  tokenizer.encode('Hello, how are you?'); // [341, 4125, 8, 140, 31, 19, 54]
});
```

To use the QnA dual encoder:
```js
// Load the model.
use.loadQnA().then(model => {
  // Embed a dictionary of a query and responses. The input to the embed method
  // needs to be in following format:
  // {
  //   queries: string[];
  //   responses: Response[];
  // }
  // queries is an array of question strings
  // responses is an array of following structure:
  // {
  //   response: string;
  //   context?: string;
  // }
  // context is optional, it provides the context string of the answer.

  const input = {
    queries: ['How are you feeling today?', 'What is captial of China?'],
    responses: [
      'I\'m not feeling very well.',
      'Beijing is the capital of China.',
      'You have five fingers on your hand.'
    ]
  };
  var scores = [];
  const embeddings = model.embed(input);
  /*
    * The output of the embed method is an object with two keys:
    * {
    *   queryEmbedding: tf.Tensor;
    *   responseEmbedding: tf.Tensor;
    * }
    * queryEmbedding is a tensor containing embeddings for all queries.
    * responseEmbedding is a tensor containing embeddings for all answers.
    * You can call `arraySync()` to retrieve the values of the tensor.
    * In this example, embed_query[0] is the embedding for the query
    * 'How are you feeling today?'
    * And embed_responses[0] is the embedding for the answer
    * 'I\'m not feeling very well.'
    */
  const scores = tf.matMul(embeddings['queryEmbedding'],
      embeddings['responseEmbedding'], false, true).dataSync();
});

```
