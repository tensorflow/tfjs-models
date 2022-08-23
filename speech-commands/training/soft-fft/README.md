# Training a TensorFlow.js model for Speech Commands Using node.js

## Preparing data for training

Before you can train your model that uses spectrogram from the browser's
WebAudio as input features, you need to download the speech-commands [data set v0.01](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz) or [data set v0.02](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

## Training the TensorFlow.js Model

The node.js training package comes with a command line tool that will assist your training. Here are the steps:
1. Prepare the node modules dependecies:

```bash
yarn
```

2. Start the CLI program:

```none
yarn start
```

Following are command supported by the CLI:

```none
  Commands:

    help [command...]           Provides help for a given command.
    exit                        Exits application.
    create_model [labels...]    create the audio model
    load_dataset all <dir>      Load all the data from the root directory by the labels
    load_dataset <dir> <label>  Load the dataset from the directory with the label
    dataset size                Show the size of the dataset
    train [epoch]               train all audio dataset
    save_model <filename>       save the audio model

```

3. You need to first create a model. For example create a model with four labels (up down left right):

```none
local@piyu~$ create up down left right

_________________________________________________________________
Layer (type)                 Output shape              Param #   
=================================================================
conv2d_Conv2D1 (Conv2D)      [null,95,39,8]            72        
_________________________________________________________________
max_pooling2d_MaxPooling2D1  [null,47,19,8]            0         
_________________________________________________________________
conv2d_Conv2D2 (Conv2D)      [null,44,18,32]           2080      
_________________________________________________________________
max_pooling2d_MaxPooling2D2  [null,22,9,32]            0         
_________________________________________________________________
conv2d_Conv2D3 (Conv2D)      [null,19,8,32]            8224      
_________________________________________________________________
max_pooling2d_MaxPooling2D3  [null,9,4,32]             0         
_________________________________________________________________
conv2d_Conv2D4 (Conv2D)      [null,6,3,32]             8224      
_________________________________________________________________
max_pooling2d_MaxPooling2D4  [null,5,1,32]             0         
_________________________________________________________________
flatten_Flatten1 (Flatten)   [null,160]                0         
_________________________________________________________________
dense_Dense1 (Dense)         [null,2000]               322000    
_________________________________________________________________
dropout_Dropout1 (Dropout)   [null,2000]               0         
_________________________________________________________________
dense_Dense2 (Dense)         [null,4]                  8004      
=================================================================
Total params: 348604
Trainable params: 348604
Non-trainable params: 0

```

4. Load the dataset. 
You can use 'load_dataset all' command to load data for all labels that is configure for the previously created model. The root directory is where you untar the dataset file to. Each label should have corresponding directory in that root directory.

```none
local@piyu~$ load_dataset all /tmp/audio/data

✔ finished loading label: up (0)
✔ finished loading label: left (2)
✔ finished loading label: down (1)
✔ finished loading label: right (3)

```

You can also load data per label using 'load' command. For example loading data for the 'up' label.

```none
local@piyu~$ load_dataset /tmp/audio/data/up up
```

5. Show the dataset stats. You can review the dataset size and shape by running 'dataset size' command.

```none
local@piyu~$ dataset size

dataset size = xs: 8534,98,40,1 ys: 8534,4
```

6. Training the model. You can also specify the epochs for the 'train' command.

```none
local@piyu~$ train 5

✔ epoch: 0, loss: 1.35054, accuracy: 0.34792, validation accuracy: 0.42740
✔ epoch: 1, loss: 1.23458, accuracy: 0.45339, validation accuracy: 0.50351
✔ epoch: 2, loss: 1.06478, accuracy: 0.55833, validation accuracy: 0.62529
✔ epoch: 3, loss: 0.88953, accuracy: 0.63073, validation accuracy: 0.68735
✔ epoch: 4, loss: 0.78241, accuracy: 0.67799, validation accuracy: 0.73770

```

7 Save the trained model. 

```none
local@piyu~$ save_model /tmp/audio_model

✔ /tmp/audio_model saved.
```

## Development