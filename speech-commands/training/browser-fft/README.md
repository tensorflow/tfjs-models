# Training a TensorFlow.js model for Speech Commands Using Browser FFT

## Preparing data for training

Before you can train your model that uses spectrogram from the browser's
WebAudio as input features, you need to convert the speech-commands
data set into a format that TensorFlow.js can ingest, by running the
data through the native WebAudio frequency analyzer (FFT) of the browser.
The following steps are involved:

1. Download the speech-commands data set from
   https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.0.1.tar.gz
   or
   https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.0.2.tar.gz
   Version 0.0.2 is a larger dataset than 0.0.1.

2. Use `prep_wavs.py` to convert the raw wav files into a binary format
   ready for FFT conversion in the browser. E.g.,

   ```sh
   python prep_wavs.py \
      --words zero,one,two,three,four,five,six,seven,eight,nine,go,stop,left,right,up,down \
      --test_split 0.15 \
      --include_noise \
       "${HOME}/ml-data/speech_commands_data" \
       "${HOME}/ml-data/speech_commands_data_converted"
   ```

   With the `--words` flag, you can specify what words to include in the
   training of the model. With the `--test_split` flag, you can specify the
   fraction of the .wav files that will be randomly drawn for testing after
   training. The `--include_noise` flag asks the script to randomly draw
   segments from the long .wav files in the '_background_noise_' folder to
   generate training (and test) examples for background noise. (N.B.: this is
   *not* about adding noise to the word examples.)
   The last two arguments point to the input and output directories,
   respectively.

   Under the output path (i.e., `speech_commands_data_converted` in this example),
   there will be two subfolders, called `train` and `test`, which hold the
   training and testing splits, respectively. Under each of `train` and `test`,
   there are subfolders with names matching the words (e.g., `zero`, `one`,
   etc.) In each of those subfolders, there will subfolders with names
   such as `0` and `1`, which contain a number of `.dat` files.

3. Run WebAudio FFT on the `.dat` files generated in step 2 in the browser.
   TODO(cais): Provide more details here.

## Training the TensorFlow.js Model in tfjs-node or tfjs-node-gpu

1. Download and extract the browser-FFT version of the speech-commands dataset:

   ```sh
   curl -fSsL https://storage.googleapis.com/learnjs-data/speech-commands/speech-commands-data-v0.02-browser.tar.gz  -o speech-commands-data-v0.02-browser.tar.gz && \
   tar xzvf speech-commands-data-v0.02-browser.tar.gz
   ```

2. Start training. First, download JavaScript dependencies using:

   ```sh
   yarn
   ```

   Then, to train the model using CPU (tfjs-node):

   ```sh
   yarn train speech-commands-data-v0.02-browser/ ./my-model/
   ```

   Or, to train the model using a GPU (tfjs-node-gpu,
   requires CUDA-enabled GPU and drivers):

   ```sh
   yarn train --gpu speech-commands-data-v0.02-browser/ ./my-model/
   ```

## Development

### Python

To run linting and tests of the Python files in this directory, use script:

```sh
./py_lint_and_test.sh
```
