/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs'], factory) :
    (factory((global.speechCommands = {}),global.tf));
}(this, (function (exports,tf) { 'use strict';

    /*! *****************************************************************************
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the Apache License, Version 2.0 (the "License"); you may not use
    this file except in compliance with the License. You may obtain a copy of the
    License at http://www.apache.org/licenses/LICENSE-2.0

    THIS CODE IS PROVIDED ON AN *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
    WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
    MERCHANTABLITY OR NON-INFRINGEMENT.

    See the Apache Version 2.0 License for specific language governing permissions
    and limitations under the License.
    ***************************************************************************** */
    /* global Reflect, Promise */

    var extendStatics = function(d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };

    function __extends(d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    }

    function __awaiter(thisArg, _arguments, P, generator) {
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    }

    function __generator(thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (_) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
        }
    }

    function __values(o) {
        var m = typeof Symbol === "function" && o[Symbol.iterator], i = 0;
        if (m) return m.call(o);
        return {
            next: function () {
                if (o && i >= o.length) o = void 0;
                return { value: o && o[i++], done: !o };
            }
        };
    }

    function __read(o, n) {
        var m = typeof Symbol === "function" && o[Symbol.iterator];
        if (!m) return o;
        var i = m.call(o), r, ar = [], e;
        try {
            while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
        }
        catch (error) { e = { error: error }; }
        finally {
            try {
                if (r && !r.done && (m = i["return"])) m.call(i);
            }
            finally { if (e) throw e.error; }
        }
        return ar;
    }

    function __spread() {
        for (var ar = [], i = 0; i < arguments.length; i++)
            ar = ar.concat(__read(arguments[i]));
        return ar;
    }

    function loadMetadataJson(url) {
        return __awaiter(this, void 0, void 0, function () {
            var HTTP_SCHEME, HTTPS_SCHEME, FILE_SCHEME, fs, content;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        HTTP_SCHEME = 'http://';
                        HTTPS_SCHEME = 'https://';
                        FILE_SCHEME = 'file://';
                        if (!(url.indexOf(HTTP_SCHEME) === 0 || url.indexOf(HTTPS_SCHEME) === 0)) return [3, 3];
                        return [4, fetch(url)];
                    case 1: return [4, (_a.sent()).json()];
                    case 2: return [2, _a.sent()];
                    case 3:
                        if (url.indexOf(FILE_SCHEME) === 0) {
                            fs = require('fs');
                            content = JSON.parse(fs.readFileSync(url.slice(FILE_SCHEME.length), { encoding: 'utf-8' }));
                            return [2, content];
                        }
                        else {
                            throw new Error("Unsupported URL scheme in metadata URL: " + url + ". " +
                                "Supported schemes are: http://, https://, and " +
                                "(node.js-only) file://");
                        }
                        _a.label = 4;
                    case 4: return [2];
                }
            });
        });
    }
    function normalize(x) {
        return tf.tidy(function () {
            var mean = tf.mean(x);
            var std = tf.sqrt(tf.mean(tf.square(tf.add(x, tf.neg(mean)))));
            return tf.div(tf.add(x, tf.neg(mean)), std);
        });
    }
    function getAudioContextConstructor() {
        return window.AudioContext || window.webkitAudioContext;
    }
    function getAudioMediaStream() {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, navigator.mediaDevices.getUserMedia({ audio: true, video: false })];
                    case 1: return [2, _a.sent()];
                }
            });
        });
    }

    var BrowserFftFeatureExtractor = (function () {
        function BrowserFftFeatureExtractor(config) {
            this.ROTATING_BUFFER_SIZE_MULTIPLIER = 2;
            if (config == null) {
                throw new Error("Required configuration object is missing for " +
                    "BrowserFftFeatureExtractor constructor");
            }
            if (config.spectrogramCallback == null) {
                throw new Error("spectrogramCallback cannot be null or undefined");
            }
            if (!(config.numFramesPerSpectrogram > 0)) {
                throw new Error("Invalid value in numFramesPerSpectrogram: " +
                    ("" + config.numFramesPerSpectrogram));
            }
            if (config.suppressionTimeMillis < 0) {
                throw new Error("Expected suppressionTimeMillis to be >= 0, " +
                    ("but got " + config.suppressionTimeMillis));
            }
            this.suppressionTimeMillis = config.suppressionTimeMillis;
            this.spectrogramCallback = config.spectrogramCallback;
            this.numFramesPerSpectrogram = config.numFramesPerSpectrogram;
            this.sampleRateHz = config.sampleRateHz || 44100;
            this.fftSize = config.fftSize || 1024;
            this.frameDurationMillis = this.fftSize / this.sampleRateHz * 1e3;
            this.columnTruncateLength = config.columnTruncateLength || this.fftSize;
            this.overlapFactor = config.overlapFactor;
            tf.util.assert(this.overlapFactor >= 0 && this.overlapFactor < 1, "Expected overlapFactor to be >= 0 and < 1, " +
                ("but got " + this.overlapFactor));
            if (this.columnTruncateLength > this.fftSize) {
                throw new Error("columnTruncateLength " + this.columnTruncateLength + " exceeds " +
                    ("fftSize (" + this.fftSize + ")."));
            }
            this.audioContextConstructor = getAudioContextConstructor();
        }
        BrowserFftFeatureExtractor.prototype.start = function (samples) {
            return __awaiter(this, void 0, void 0, function () {
                var _a, streamSource, rotatingBufferSize, period;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            if (this.frameIntervalTask != null) {
                                throw new Error('Cannot start already-started BrowserFftFeatureExtractor');
                            }
                            _a = this;
                            return [4, getAudioMediaStream()];
                        case 1:
                            _a.stream = _b.sent();
                            this.audioContext = new this.audioContextConstructor();
                            if (this.audioContext.sampleRate !== this.sampleRateHz) {
                                console.warn("Mismatch in sampling rate: " +
                                    ("Expected: " + this.sampleRateHz + "; ") +
                                    ("Actual: " + this.audioContext.sampleRate));
                            }
                            streamSource = this.audioContext.createMediaStreamSource(this.stream);
                            this.analyser = this.audioContext.createAnalyser();
                            this.analyser.fftSize = this.fftSize * 2;
                            this.analyser.smoothingTimeConstant = 0.0;
                            streamSource.connect(this.analyser);
                            this.freqData = new Float32Array(this.fftSize);
                            this.rotatingBufferNumFrames =
                                this.numFramesPerSpectrogram * this.ROTATING_BUFFER_SIZE_MULTIPLIER;
                            rotatingBufferSize = this.columnTruncateLength * this.rotatingBufferNumFrames;
                            this.rotatingBuffer = new Float32Array(rotatingBufferSize);
                            this.frameCount = 0;
                            period = Math.max(1, Math.round(this.numFramesPerSpectrogram * (1 - this.overlapFactor)));
                            this.tracker = new Tracker(period, Math.round(this.suppressionTimeMillis / this.frameDurationMillis));
                            this.frameIntervalTask = setInterval(this.onAudioFrame.bind(this), this.fftSize / this.sampleRateHz * 1e3);
                            return [2];
                    }
                });
            });
        };
        BrowserFftFeatureExtractor.prototype.onAudioFrame = function () {
            return __awaiter(this, void 0, void 0, function () {
                var freqDataSlice, bufferPos, shouldFire, freqData, inputTensor, shouldRest;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            this.analyser.getFloatFrequencyData(this.freqData);
                            if (this.freqData[0] === -Infinity) {
                                return [2];
                            }
                            freqDataSlice = this.freqData.slice(0, this.columnTruncateLength);
                            bufferPos = this.frameCount % this.rotatingBufferNumFrames;
                            this.rotatingBuffer.set(freqDataSlice, bufferPos * this.columnTruncateLength);
                            this.frameCount++;
                            shouldFire = this.tracker.tick();
                            if (!shouldFire) return [3, 2];
                            freqData = getFrequencyDataFromRotatingBuffer(this.rotatingBuffer, this.numFramesPerSpectrogram, this.columnTruncateLength, this.frameCount - this.numFramesPerSpectrogram);
                            inputTensor = getInputTensorFromFrequencyData(freqData, this.numFramesPerSpectrogram, this.columnTruncateLength);
                            return [4, this.spectrogramCallback(inputTensor)];
                        case 1:
                            shouldRest = _a.sent();
                            if (shouldRest) {
                                this.tracker.suppress();
                            }
                            inputTensor.dispose();
                            _a.label = 2;
                        case 2: return [2];
                    }
                });
            });
        };
        BrowserFftFeatureExtractor.prototype.stop = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    if (this.frameIntervalTask == null) {
                        throw new Error('Cannot stop because there is no ongoing streaming activity.');
                    }
                    clearInterval(this.frameIntervalTask);
                    this.frameIntervalTask = null;
                    this.analyser.disconnect();
                    this.audioContext.close();
                    return [2];
                });
            });
        };
        BrowserFftFeatureExtractor.prototype.setConfig = function (params) {
            throw new Error('setConfig() is not implemented for BrowserFftFeatureExtractor.');
        };
        BrowserFftFeatureExtractor.prototype.getFeatures = function () {
            throw new Error('getFeatures() is not implemented for ' +
                'BrowserFftFeatureExtractor. Use the spectrogramCallback ' +
                'field of the constructor config instead.');
        };
        return BrowserFftFeatureExtractor;
    }());
    function getFrequencyDataFromRotatingBuffer(rotatingBuffer, numFrames, fftLength, frameCount) {
        var size = numFrames * fftLength;
        var freqData = new Float32Array(size);
        var rotatingBufferSize = rotatingBuffer.length;
        var rotatingBufferNumFrames = rotatingBufferSize / fftLength;
        while (frameCount < 0) {
            frameCount += rotatingBufferNumFrames;
        }
        var indexBegin = (frameCount % rotatingBufferNumFrames) * fftLength;
        var indexEnd = indexBegin + size;
        for (var i = indexBegin; i < indexEnd; ++i) {
            freqData[i - indexBegin] = rotatingBuffer[i % rotatingBufferSize];
        }
        return freqData;
    }
    function getInputTensorFromFrequencyData(freqData, numFrames, fftLength, toNormalize) {
        if (toNormalize === void 0) { toNormalize = true; }
        return tf.tidy(function () {
            var size = freqData.length;
            var tensorBuffer = tf.buffer([size]);
            for (var i = 0; i < freqData.length; ++i) {
                tensorBuffer.set(freqData[i], i);
            }
            var output = tensorBuffer.toTensor().reshape([1, numFrames, fftLength, 1]);
            return toNormalize ? normalize(output) : output;
        });
    }
    var Tracker = (function () {
        function Tracker(period, suppressionPeriod) {
            this.period = period;
            this.suppressionTime = suppressionPeriod == null ? 0 : suppressionPeriod;
            this.counter = 0;
            tf.util.assert(this.period > 0, "Expected period to be positive, but got " + this.period);
        }
        Tracker.prototype.tick = function () {
            this.counter++;
            var shouldFire = (this.counter % this.period === 0) &&
                (this.suppressionOnset == null ||
                    this.counter - this.suppressionOnset > this.suppressionTime);
            return shouldFire;
        };
        Tracker.prototype.suppress = function () {
            this.suppressionOnset = this.counter;
        };
        return Tracker;
    }());

    var version = '0.1.3';

    var BACKGROUND_NOISE_TAG = '_background_noise_';
    var UNKNOWN_TAG = '_unknown_';
    var streaming = false;
    var BrowserFftSpeechCommandRecognizer = (function () {
        function BrowserFftSpeechCommandRecognizer(vocabulary, modelURL, metadataURL) {
            this.MODEL_URL_PREFIX = "https://storage.googleapis.com/tfjs-speech-commands-models/v" + version + "/browser_fft";
            this.SAMPLE_RATE_HZ = 44100;
            this.FFT_SIZE = 1024;
            this.DEFAULT_SUPPRESSION_TIME_MILLIS = 0;
            this.transferRecognizers = {};
            tf.util.assert(modelURL == null && metadataURL == null ||
                modelURL != null && metadataURL != null, "modelURL and metadataURL must be both provided or " +
                "both not provided.");
            if (modelURL == null) {
                if (vocabulary == null) {
                    vocabulary = BrowserFftSpeechCommandRecognizer.DEFAULT_VOCABULARY_NAME;
                }
                else {
                    tf.util.assert(BrowserFftSpeechCommandRecognizer.VALID_VOCABULARY_NAMES.indexOf(vocabulary) !== -1, "Invalid vocabulary name: '" + vocabulary + "'");
                }
                this.vocabulary = vocabulary;
                this.modelURL = this.MODEL_URL_PREFIX + "/" + this.vocabulary + "/model.json";
                this.metadataURL =
                    this.MODEL_URL_PREFIX + "/" + this.vocabulary + "/metadata.json";
            }
            else {
                tf.util.assert(vocabulary == null, "vocabulary name must be null or undefined when modelURL is " +
                    "provided");
                this.modelURL = modelURL;
                this.metadataURL = metadataURL;
            }
            this.parameters = {
                sampleRateHz: this.SAMPLE_RATE_HZ,
                fftSize: this.FFT_SIZE
            };
        }
        BrowserFftSpeechCommandRecognizer.prototype.startStreaming = function (callback, config) {
            return __awaiter(this, void 0, void 0, function () {
                var probabilityThreshold, invokeCallbackOnNoiseAndUnknown, overlapFactor, spectrogramCallback, suppressionTimeMillis;
                var _this = this;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (streaming) {
                                throw new Error('Cannot start streaming again when streaming is ongoing.');
                            }
                            return [4, this.ensureModelLoaded()];
                        case 1:
                            _a.sent();
                            if (config == null) {
                                config = {};
                            }
                            probabilityThreshold = config.probabilityThreshold == null ? 0 : config.probabilityThreshold;
                            if (config.includeEmbedding) {
                                probabilityThreshold = 0;
                            }
                            tf.util.assert(probabilityThreshold >= 0 && probabilityThreshold <= 1, "Invalid probabilityThreshold value: " + probabilityThreshold);
                            invokeCallbackOnNoiseAndUnknown = config.invokeCallbackOnNoiseAndUnknown == null ?
                                false :
                                config.invokeCallbackOnNoiseAndUnknown;
                            if (config.includeEmbedding) {
                                invokeCallbackOnNoiseAndUnknown = true;
                            }
                            if (config.suppressionTimeMillis < 0) {
                                throw new Error("suppressionTimeMillis is expected to be >= 0, " +
                                    ("but got " + config.suppressionTimeMillis));
                            }
                            overlapFactor = config.overlapFactor == null ? 0.5 : config.overlapFactor;
                            tf.util.assert(overlapFactor >= 0 && overlapFactor < 1, "Expected overlapFactor to be >= 0 and < 1, but got " + overlapFactor);
                            spectrogramCallback = function (x) { return __awaiter(_this, void 0, void 0, function () {
                                var _a, y, embedding, scores, maxIndexTensor, maxIndex, maxScore, spectrogram, _b, wordDetected;
                                return __generator(this, function (_c) {
                                    switch (_c.label) {
                                        case 0: return [4, this.ensureModelWithEmbeddingOutputCreated()];
                                        case 1:
                                            _c.sent();
                                            if (!config.includeEmbedding) return [3, 3];
                                            return [4, this.ensureModelWithEmbeddingOutputCreated()];
                                        case 2:
                                            _c.sent();
                                            _a = __read(this.modelWithEmbeddingOutput.predict(x), 2), y = _a[0], embedding = _a[1];
                                            return [3, 4];
                                        case 3:
                                            y = this.model.predict(x);
                                            _c.label = 4;
                                        case 4: return [4, y.data()];
                                        case 5:
                                            scores = _c.sent();
                                            maxIndexTensor = y.argMax(-1);
                                            return [4, maxIndexTensor.data()];
                                        case 6:
                                            maxIndex = (_c.sent())[0];
                                            maxScore = Math.max.apply(Math, __spread(scores));
                                            tf.dispose([y, maxIndexTensor]);
                                            if (!(maxScore < probabilityThreshold)) return [3, 7];
                                            return [2, false];
                                        case 7:
                                            spectrogram = undefined;
                                            if (!config.includeSpectrogram) return [3, 9];
                                            _b = {};
                                            return [4, x.data()];
                                        case 8:
                                            spectrogram = (_b.data = (_c.sent()),
                                                _b.frameSize = this.nonBatchInputShape[1],
                                                _b);
                                            _c.label = 9;
                                        case 9:
                                            wordDetected = true;
                                            if (!invokeCallbackOnNoiseAndUnknown) {
                                                if (this.words[maxIndex] === BACKGROUND_NOISE_TAG ||
                                                    this.words[maxIndex] === UNKNOWN_TAG) {
                                                    wordDetected = false;
                                                }
                                            }
                                            if (wordDetected) {
                                                callback({ scores: scores, spectrogram: spectrogram, embedding: embedding });
                                            }
                                            return [2, wordDetected];
                                    }
                                });
                            }); };
                            suppressionTimeMillis = config.suppressionTimeMillis == null ?
                                this.DEFAULT_SUPPRESSION_TIME_MILLIS :
                                config.suppressionTimeMillis;
                            this.audioDataExtractor = new BrowserFftFeatureExtractor({
                                sampleRateHz: this.parameters.sampleRateHz,
                                numFramesPerSpectrogram: this.nonBatchInputShape[0],
                                columnTruncateLength: this.nonBatchInputShape[1],
                                suppressionTimeMillis: suppressionTimeMillis,
                                spectrogramCallback: spectrogramCallback,
                                overlapFactor: overlapFactor
                            });
                            return [4, this.audioDataExtractor.start()];
                        case 2:
                            _a.sent();
                            streaming = true;
                            return [2];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.ensureModelLoaded = function () {
            return __awaiter(this, void 0, void 0, function () {
                var model, outputShape, frameDurationMillis, numFrames;
                var _this = this;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (this.model != null) {
                                return [2];
                            }
                            return [4, this.ensureMetadataLoaded()];
                        case 1:
                            _a.sent();
                            return [4, tf.loadModel(this.modelURL)];
                        case 2:
                            model = _a.sent();
                            if (model.inputs.length !== 1) {
                                throw new Error("Expected model to have 1 input, but got a model with " +
                                    (model.inputs.length + " inputs"));
                            }
                            if (model.inputs[0].shape.length !== 4) {
                                throw new Error("Expected model to have an input shape of rank 4, " +
                                    ("but got an input shape of rank " + model.inputs[0].shape.length));
                            }
                            if (model.inputs[0].shape[3] !== 1) {
                                throw new Error("Expected model to have an input shape with 1 as the last " +
                                    "dimension, but got input shape" +
                                    (JSON.stringify(model.inputs[0].shape[3]) + "}"));
                            }
                            outputShape = model.outputShape;
                            if (outputShape.length !== 2) {
                                throw new Error("Expected loaded model to have an output shape of rank 2," +
                                    ("but received shape " + JSON.stringify(outputShape)));
                            }
                            if (outputShape[1] !== this.words.length) {
                                throw new Error("Mismatch between the last dimension of model's output shape " +
                                    ("(" + outputShape[1] + ") and number of words ") +
                                    ("(" + this.words.length + ")."));
                            }
                            this.model = model;
                            this.freezeModel();
                            this.nonBatchInputShape =
                                model.inputs[0].shape.slice(1);
                            this.elementsPerExample = 1;
                            model.inputs[0].shape.slice(1).forEach(function (dimSize) { return _this.elementsPerExample *= dimSize; });
                            this.warmUpModel();
                            frameDurationMillis = this.parameters.fftSize / this.parameters.sampleRateHz * 1e3;
                            numFrames = model.inputs[0].shape[1];
                            this.parameters.spectrogramDurationMillis = numFrames * frameDurationMillis;
                            return [2];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.ensureModelWithEmbeddingOutputCreated = function () {
            return __awaiter(this, void 0, void 0, function () {
                var secondLastDenseLayer, i;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (this.modelWithEmbeddingOutput != null) {
                                return [2];
                            }
                            return [4, this.ensureModelLoaded()];
                        case 1:
                            _a.sent();
                            for (i = this.model.layers.length - 2; i >= 0; --i) {
                                if (this.model.layers[i].getClassName() === 'Dense') {
                                    secondLastDenseLayer = this.model.layers[i];
                                    break;
                                }
                            }
                            if (secondLastDenseLayer == null) {
                                throw new Error('Failed to find second last dense layer in the original model.');
                            }
                            this.modelWithEmbeddingOutput = tf.model({
                                inputs: this.model.inputs,
                                outputs: [
                                    this.model.outputs[0], secondLastDenseLayer.output
                                ]
                            });
                            return [2];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.warmUpModel = function () {
            var _this = this;
            tf.tidy(function () {
                var x = tf.zeros([1].concat(_this.nonBatchInputShape));
                for (var i = 0; i < 3; ++i) {
                    _this.model.predict(x);
                }
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.ensureMetadataLoaded = function () {
            return __awaiter(this, void 0, void 0, function () {
                var metadataJSON;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (this.words != null) {
                                return [2];
                            }
                            return [4, loadMetadataJson(this.metadataURL)];
                        case 1:
                            metadataJSON = _a.sent();
                            this.words = metadataJSON.words;
                            return [2];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.stopStreaming = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            if (!streaming) {
                                throw new Error('Cannot stop streaming when streaming is not ongoing.');
                            }
                            return [4, this.audioDataExtractor.stop()];
                        case 1:
                            _a.sent();
                            streaming = false;
                            return [2];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.isStreaming = function () {
            return streaming;
        };
        BrowserFftSpeechCommandRecognizer.prototype.wordLabels = function () {
            return this.words;
        };
        BrowserFftSpeechCommandRecognizer.prototype.params = function () {
            return this.parameters;
        };
        BrowserFftSpeechCommandRecognizer.prototype.modelInputShape = function () {
            if (this.model == null) {
                throw new Error('Model has not been loaded yet. Load model by calling ' +
                    'ensureModelLoaded(), recognizer(), or startStreaming().');
            }
            return this.model.inputs[0].shape;
        };
        BrowserFftSpeechCommandRecognizer.prototype.recognize = function (input, config) {
            return __awaiter(this, void 0, void 0, function () {
                var spectrogramData, numExamples, inputTensor, outTensor, output, outAndEmbedding, _a, unstacked, scorePromises, _b;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            if (config == null) {
                                config = {};
                            }
                            return [4, this.ensureModelLoaded()];
                        case 1:
                            _c.sent();
                            if (!(input == null)) return [3, 3];
                            return [4, this.recognizeOnline()];
                        case 2:
                            spectrogramData = _c.sent();
                            input = spectrogramData.data;
                            _c.label = 3;
                        case 3:
                            if (input instanceof tf.Tensor) {
                                this.checkInputTensorShape(input);
                                inputTensor = input;
                                numExamples = input.shape[0];
                            }
                            else {
                                input = input;
                                if (input.length % this.elementsPerExample) {
                                    throw new Error("The length of the input Float32Array " + input.length + " " +
                                        "is not divisible by the number of tensor elements per " +
                                        ("per example expected by the model " + this.elementsPerExample + "."));
                                }
                                numExamples = input.length / this.elementsPerExample;
                                inputTensor = tf.tensor4d(input, [
                                    numExamples
                                ].concat(this.nonBatchInputShape));
                            }
                            output = { scores: null };
                            if (!config.includeEmbedding) return [3, 5];
                            return [4, this.ensureModelWithEmbeddingOutputCreated()];
                        case 4:
                            _c.sent();
                            outAndEmbedding = this.modelWithEmbeddingOutput.predict(inputTensor);
                            outTensor = outAndEmbedding[0];
                            output.embedding = outAndEmbedding[1];
                            return [3, 6];
                        case 5:
                            outTensor = this.model.predict(inputTensor);
                            _c.label = 6;
                        case 6:
                            if (!(numExamples === 1)) return [3, 8];
                            _a = output;
                            return [4, outTensor.data()];
                        case 7:
                            _a.scores = (_c.sent());
                            return [3, 10];
                        case 8:
                            unstacked = tf.unstack(outTensor);
                            scorePromises = unstacked.map(function (item) { return item.data(); });
                            _b = output;
                            return [4, Promise.all(scorePromises)];
                        case 9:
                            _b.scores = (_c.sent());
                            tf.dispose(unstacked);
                            _c.label = 10;
                        case 10: return [2, output];
                    }
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.recognizeOnline = function () {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    return [2, new Promise(function (resolve, reject) {
                            var spectrogramCallback = function (x) { return __awaiter(_this, void 0, void 0, function () {
                                var _a, _b;
                                return __generator(this, function (_c) {
                                    switch (_c.label) {
                                        case 0:
                                            _a = resolve;
                                            _b = {};
                                            return [4, x.data()];
                                        case 1:
                                            _a.apply(void 0, [(_b.data = (_c.sent()),
                                                    _b.frameSize = this.nonBatchInputShape[1],
                                                    _b)]);
                                            return [2, false];
                                    }
                                });
                            }); };
                            _this.audioDataExtractor = new BrowserFftFeatureExtractor({
                                sampleRateHz: _this.parameters.sampleRateHz,
                                numFramesPerSpectrogram: _this.nonBatchInputShape[0],
                                columnTruncateLength: _this.nonBatchInputShape[1],
                                suppressionTimeMillis: 0,
                                spectrogramCallback: spectrogramCallback,
                                overlapFactor: 0
                            });
                            _this.audioDataExtractor.start();
                        })];
                });
            });
        };
        BrowserFftSpeechCommandRecognizer.prototype.createTransfer = function (name) {
            if (this.model == null) {
                throw new Error('Model has not been loaded yet. Load model by calling ' +
                    'ensureModelLoaded(), recognizer(), or startStreaming().');
            }
            tf.util.assert(name != null && typeof name === 'string' && name.length > 1, "Expected the name for a transfer-learning recognized to be a " +
                ("non-empty string, but got " + JSON.stringify(name)));
            tf.util.assert(this.transferRecognizers[name] == null, "There is already a transfer-learning model named '" + name + "'");
            var transfer = new TransferBrowserFftSpeechCommandRecognizer(name, this.parameters, this.model);
            this.transferRecognizers[name] = transfer;
            return transfer;
        };
        BrowserFftSpeechCommandRecognizer.prototype.freezeModel = function () {
            var e_1, _a;
            try {
                for (var _b = __values(this.model.layers), _c = _b.next(); !_c.done; _c = _b.next()) {
                    var layer = _c.value;
                    layer.trainable = false;
                }
            }
            catch (e_1_1) { e_1 = { error: e_1_1 }; }
            finally {
                try {
                    if (_c && !_c.done && (_a = _b.return)) _a.call(_b);
                }
                finally { if (e_1) throw e_1.error; }
            }
        };
        BrowserFftSpeechCommandRecognizer.prototype.checkInputTensorShape = function (input) {
            var expectedRank = this.model.inputs[0].shape.length;
            if (input.shape.length !== expectedRank) {
                throw new Error("Expected input Tensor to have rank " + expectedRank + ", " +
                    ("but got rank " + input.shape.length + " that differs "));
            }
            var nonBatchedShape = input.shape.slice(1);
            var expectedNonBatchShape = this.model.inputs[0].shape.slice(1);
            if (!tf.util.arraysEqual(nonBatchedShape, expectedNonBatchShape)) {
                throw new Error("Expected input to have shape [null," + expectedNonBatchShape + "], " +
                    ("but got shape [null," + nonBatchedShape + "]"));
            }
        };
        BrowserFftSpeechCommandRecognizer.VALID_VOCABULARY_NAMES = ['18w', 'directional4w'];
        BrowserFftSpeechCommandRecognizer.DEFAULT_VOCABULARY_NAME = '18w';
        return BrowserFftSpeechCommandRecognizer;
    }());
    var TransferBrowserFftSpeechCommandRecognizer = (function (_super) {
        __extends(TransferBrowserFftSpeechCommandRecognizer, _super);
        function TransferBrowserFftSpeechCommandRecognizer(name, parameters, baseModel) {
            var _this = _super.call(this) || this;
            _this.name = name;
            _this.parameters = parameters;
            _this.baseModel = baseModel;
            tf.util.assert(name != null && typeof name === 'string' && name.length > 0, "The name of a transfer model must be a non-empty string, " +
                ("but got " + JSON.stringify(name)));
            _this.nonBatchInputShape =
                _this.baseModel.inputs[0].shape.slice(1);
            _this.words = [];
            return _this;
        }
        TransferBrowserFftSpeechCommandRecognizer.prototype.collectExample = function (word) {
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                return __generator(this, function (_a) {
                    tf.util.assert(!streaming, 'Cannot start collection of transfer-learning example because ' +
                        'a streaming recognition or transfer-learning example collection ' +
                        'is ongoing');
                    tf.util.assert(word != null && typeof word === 'string' && word.length > 0, "Must provide a non-empty string when collecting transfer-" +
                        "learning example");
                    streaming = true;
                    return [2, new Promise(function (resolve, reject) {
                            var spectrogramCallback = function (x) { return __awaiter(_this, void 0, void 0, function () {
                                var _a, _b;
                                return __generator(this, function (_c) {
                                    switch (_c.label) {
                                        case 0:
                                            if (this.transferExamples == null) {
                                                this.transferExamples = {};
                                            }
                                            if (this.transferExamples[word] == null) {
                                                this.transferExamples[word] = [];
                                            }
                                            this.transferExamples[word].push(x.clone());
                                            return [4, this.audioDataExtractor.stop()];
                                        case 1:
                                            _c.sent();
                                            streaming = false;
                                            this.collateTransferWords();
                                            _a = resolve;
                                            _b = {};
                                            return [4, x.data()];
                                        case 2:
                                            _a.apply(void 0, [(_b.data = (_c.sent()),
                                                    _b.frameSize = this.nonBatchInputShape[1],
                                                    _b)]);
                                            return [2, false];
                                    }
                                });
                            }); };
                            _this.audioDataExtractor = new BrowserFftFeatureExtractor({
                                sampleRateHz: _this.parameters.sampleRateHz,
                                numFramesPerSpectrogram: _this.nonBatchInputShape[0],
                                columnTruncateLength: _this.nonBatchInputShape[1],
                                suppressionTimeMillis: 0,
                                spectrogramCallback: spectrogramCallback,
                                overlapFactor: 0
                            });
                            _this.audioDataExtractor.start();
                        })];
                });
            });
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.clearExamples = function () {
            tf.util.assert(this.words != null && this.words.length > 0 &&
                this.transferExamples != null, "No transfer learning examples exist for model name " + this.name);
            tf.dispose(this.transferExamples);
            this.transferExamples = null;
            this.words = null;
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.countExamples = function () {
            if (this.transferExamples == null) {
                throw new Error("No examples have been collected for transfer-learning model " +
                    ("named '" + this.name + "' yet."));
            }
            var counts = {};
            for (var word in this.transferExamples) {
                counts[word] = this.transferExamples[word].length;
            }
            return counts;
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.collateTransferWords = function () {
            this.words = Object.keys(this.transferExamples).sort();
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.collectTransferDataAsTensors = function (modelName) {
            var _this = this;
            tf.util.assert(this.words != null && this.words.length > 0, "No word example is available for tranfer-learning model of name " +
                modelName);
            return tf.tidy(function () {
                var xTensors = [];
                var targetIndices = [];
                _this.words.forEach(function (word, i) {
                    _this.transferExamples[word].forEach(function (wordTensor) {
                        xTensors.push(wordTensor);
                        targetIndices.push(i);
                    });
                });
                return {
                    xs: tf.concat(xTensors, 0),
                    ys: tf.oneHot(tf.tensor1d(targetIndices, 'int32'), Object.keys(_this.words).length)
                };
            });
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.train = function (config) {
            return __awaiter(this, void 0, void 0, function () {
                var optimizer, _a, xs, ys, epochs, validationSplit, history_1, err_1;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            tf.util.assert(this.words != null && this.words.length > 0, "Cannot train transfer-learning model '" + this.name + "' because no " +
                                "transfer learning example has been collected.");
                            tf.util.assert(this.words.length > 1, "Cannot train transfer-learning model '" + this.name + "' because only " +
                                ("1 word label ('" + JSON.stringify(this.words) + "') ") +
                                "has been collected for transfer learning. Requires at least 2.");
                            if (config == null) {
                                config = {};
                            }
                            if (this.model == null) {
                                this.createTransferModelFromBaseModel();
                            }
                            optimizer = config.optimizer || 'sgd';
                            this.model.compile({ loss: 'categoricalCrossentropy', optimizer: optimizer, metrics: ['acc'] });
                            _a = this.collectTransferDataAsTensors(), xs = _a.xs, ys = _a.ys;
                            epochs = config.epochs == null ? 20 : config.epochs;
                            validationSplit = config.validationSplit == null ? 0 : config.validationSplit;
                            _b.label = 1;
                        case 1:
                            _b.trys.push([1, 3, , 4]);
                            return [4, this.model.fit(xs, ys, {
                                    epochs: epochs,
                                    validationSplit: validationSplit,
                                    batchSize: config.batchSize,
                                    callbacks: config.callback == null ? null : [config.callback]
                                })];
                        case 2:
                            history_1 = _b.sent();
                            tf.dispose([xs, ys]);
                            return [2, history_1];
                        case 3:
                            err_1 = _b.sent();
                            tf.dispose([xs, ys]);
                            this.model = null;
                            return [2, null];
                        case 4: return [2];
                    }
                });
            });
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.createTransferModelFromBaseModel = function () {
            tf.util.assert(this.words != null, "No word example is available for tranfer-learning model of name " +
                this.name);
            var layers = this.baseModel.layers;
            var layerIndex = layers.length - 2;
            while (layerIndex >= 0) {
                if (layers[layerIndex].getClassName().toLowerCase() === 'dense') {
                    break;
                }
                layerIndex--;
            }
            if (layerIndex < 0) {
                throw new Error('Cannot find a hidden dense layer in the base model.');
            }
            var beheadedBaseOutput = layers[layerIndex].output;
            this.transferHead = tf.sequential();
            this.transferHead.add(tf.layers.dense({
                units: this.words.length,
                activation: 'softmax',
                inputShape: beheadedBaseOutput.shape.slice(1)
            }));
            var transferOutput = this.transferHead.apply(beheadedBaseOutput);
            this.model =
                tf.model({ inputs: this.baseModel.inputs, outputs: transferOutput });
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.modelInputShape = function () {
            return this.baseModel.inputs[0].shape;
        };
        TransferBrowserFftSpeechCommandRecognizer.prototype.createTransfer = function (name) {
            throw new Error('Creating transfer-learned recognizer from a transfer-learned ' +
                'recognizer is not supported.');
        };
        return TransferBrowserFftSpeechCommandRecognizer;
    }(BrowserFftSpeechCommandRecognizer));

    function create(fftType, vocabulary, customModelURL, customMetadataURL) {
        tf.util.assert(customModelURL == null && customMetadataURL == null ||
            customModelURL != null && customMetadataURL != null, "customModelURL and customMetadataURL must be both provided or " +
            "both not provided.");
        if (customModelURL != null) {
            tf.util.assert(vocabulary == null, "vocabulary name must be null or undefined when modelURL is provided");
        }
        if (fftType === 'BROWSER_FFT') {
            return new BrowserFftSpeechCommandRecognizer(vocabulary, customModelURL, customMetadataURL);
        }
        else if (fftType === 'SOFT_FFT') {
            throw new Error('SOFT_FFT SpeechCommandRecognizer has not been implemented yet.');
        }
        else {
            throw new Error("Invalid fftType: '" + fftType + "'");
        }
    }

    exports.create = create;
    exports.BACKGROUND_NOISE_TAG = BACKGROUND_NOISE_TAG;
    exports.UNKNOWN_TAG = UNKNOWN_TAG;
    exports.version = version;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=speech-commands.js.map
