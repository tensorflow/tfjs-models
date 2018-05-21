// @tensorflow/tfjs-models Copyright 2018 Google
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs'], factory) :
    (factory((global.posenet = {}),global.tf));
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
                if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [0, t.value];
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

    var partNames = [
        'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
        'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
        'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
    ];
    var NUM_KEYPOINTS = partNames.length;
    var partIds = partNames.reduce(function (result, jointName, i) {
        result[jointName] = i;
        return result;
    }, {});
    var connectedPartNames = [
        ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
        ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
        ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
        ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
        ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
    ];
    var connectedPartIndeces = connectedPartNames.map(function (_a) {
        var jointNameA = _a[0], jointNameB = _a[1];
        return ([partIds[jointNameA], partIds[jointNameB]]);
    });

    function eitherPointDoesntMeetConfidence(a, b, minConfidence) {
        return (a < minConfidence || b < minConfidence);
    }
    function getAdjacentKeyPoints(keypoints, minConfidence) {
        return connectedPartIndeces.reduce(function (result, _a) {
            var leftJoint = _a[0], rightJoint = _a[1];
            if (eitherPointDoesntMeetConfidence(keypoints[leftJoint].score, keypoints[rightJoint].score, minConfidence)) {
                return result;
            }
            result.push([keypoints[leftJoint], keypoints[rightJoint]]);
            return result;
        }, []);
    }
    var NEGATIVE_INFINITY = Number.NEGATIVE_INFINITY, POSITIVE_INFINITY = Number.POSITIVE_INFINITY;
    function getBoundingBox(keypoints) {
        return keypoints.reduce(function (_a, _b) {
            var maxX = _a.maxX, maxY = _a.maxY, minX = _a.minX, minY = _a.minY;
            var _c = _b.position, x = _c.x, y = _c.y;
            return {
                maxX: Math.max(maxX, x),
                maxY: Math.max(maxY, y),
                minX: Math.min(minX, x),
                minY: Math.min(minY, y)
            };
        }, {
            maxX: NEGATIVE_INFINITY,
            maxY: NEGATIVE_INFINITY,
            minX: POSITIVE_INFINITY,
            minY: POSITIVE_INFINITY
        });
    }
    function getBoundingBoxPoints(keypoints) {
        var _a = getBoundingBox(keypoints), minX = _a.minX, minY = _a.minY, maxX = _a.maxX, maxY = _a.maxY;
        return [
            { x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY },
            { x: minX, y: maxY }
        ];
    }
    function toTensorBuffer(tensor, type) {
        if (type === void 0) { type = 'float32'; }
        return __awaiter(this, void 0, void 0, function () {
            var tensorData;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4, tensor.data()];
                    case 1:
                        tensorData = _a.sent();
                        return [2, new tf.TensorBuffer(tensor.shape, type, tensorData)];
                }
            });
        });
    }
    function toTensorBuffers3D(tensors) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                return [2, Promise.all(tensors.map(function (tensor) { return toTensorBuffer(tensor, 'float32'); }))];
            });
        });
    }
    function scalePose(pose, scaleX, scaleY) {
        return {
            score: pose.score,
            keypoints: pose.keypoints.map(function (_a) {
                var score = _a.score, part = _a.part, position = _a.position;
                return ({
                    score: score,
                    part: part,
                    position: { x: position.x * scaleX, y: position.y * scaleY }
                });
            })
        };
    }
    function scalePoses(poses, scaleY, scaleX) {
        if (scaleX === 1 && scaleY === 1) {
            return poses;
        }
        return poses.map(function (pose) { return scalePose(pose, scaleX, scaleY); });
    }
    function getValidResolution(imageScaleFactor, inputDimension, outputStride) {
        var evenResolution = inputDimension * imageScaleFactor - 1;
        return evenResolution - (evenResolution % outputStride) + 1;
    }

    function half(k) {
        return Math.floor(k / 2);
    }
    var MaxHeap = (function () {
        function MaxHeap(maxSize, getElementValue) {
            this.priorityQueue = new Array(maxSize);
            this.numberOfElements = -1;
            this.getElementValue = getElementValue;
        }
        MaxHeap.prototype.enqueue = function (x) {
            this.priorityQueue[++this.numberOfElements] = x;
            this.swim(this.numberOfElements);
        };
        MaxHeap.prototype.dequeue = function () {
            var max = this.priorityQueue[0];
            this.exchange(0, this.numberOfElements--);
            this.sink(0);
            this.priorityQueue[this.numberOfElements + 1] = null;
            return max;
        };
        MaxHeap.prototype.empty = function () {
            return this.numberOfElements === -1;
        };
        MaxHeap.prototype.size = function () {
            return this.numberOfElements + 1;
        };
        MaxHeap.prototype.all = function () {
            return this.priorityQueue.slice(0, this.numberOfElements + 1);
        };
        MaxHeap.prototype.max = function () {
            return this.priorityQueue[0];
        };
        MaxHeap.prototype.swim = function (k) {
            while (k > 0 && this.less(half(k), k)) {
                this.exchange(k, half(k));
                k = half(k);
            }
        };
        MaxHeap.prototype.sink = function (k) {
            while (2 * k <= this.numberOfElements) {
                var j = 2 * k;
                if (j < this.numberOfElements && this.less(j, j + 1)) {
                    j++;
                }
                if (!this.less(k, j)) {
                    break;
                }
                this.exchange(k, j);
                k = j;
            }
        };
        MaxHeap.prototype.getValueAt = function (i) {
            return this.getElementValue(this.priorityQueue[i]);
        };
        MaxHeap.prototype.less = function (i, j) {
            return this.getValueAt(i) < this.getValueAt(j);
        };
        MaxHeap.prototype.exchange = function (i, j) {
            var t = this.priorityQueue[i];
            this.priorityQueue[i] = this.priorityQueue[j];
            this.priorityQueue[j] = t;
        };
        return MaxHeap;
    }());

    function scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores) {
        var _a = scores.shape, height = _a[0], width = _a[1];
        var localMaximum = true;
        var yStart = Math.max(heatmapY - localMaximumRadius, 0);
        var yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
        for (var yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            var xStart = Math.max(heatmapX - localMaximumRadius, 0);
            var xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
            for (var xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (scores.get(yCurrent, xCurrent, keypointId) > score) {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum) {
                break;
            }
        }
        return localMaximum;
    }
    function buildPartWithScoreQueue(scoreThreshold, localMaximumRadius, scores) {
        var _a = scores.shape, height = _a[0], width = _a[1], numKeypoints = _a[2];
        var queue = new MaxHeap(height * width * numKeypoints, function (_a) {
            var score = _a.score;
            return score;
        });
        for (var heatmapY = 0; heatmapY < height; ++heatmapY) {
            for (var heatmapX = 0; heatmapX < width; ++heatmapX) {
                for (var keypointId = 0; keypointId < numKeypoints; ++keypointId) {
                    var score = scores.get(heatmapY, heatmapX, keypointId);
                    if (score < scoreThreshold) {
                        continue;
                    }
                    if (scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        queue.enqueue({ score: score, part: { heatmapY: heatmapY, heatmapX: heatmapX, id: keypointId } });
                    }
                }
            }
        }
        return queue;
    }

    function getOffsetPoint(y, x, keypoint, offsets) {
        return {
            y: offsets.get(y, x, keypoint),
            x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
        };
    }
    function getImageCoords(part, outputStride, offsets) {
        var heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
        var _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets), y = _a.y, x = _a.x;
        return {
            x: part.heatmapX * outputStride + x,
            y: part.heatmapY * outputStride + y
        };
    }
    function clamp(a, min, max) {
        if (a < min) {
            return min;
        }
        if (a > max) {
            return max;
        }
        return a;
    }
    function squaredDistance(y1, x1, y2, x2) {
        var dy = y2 - y1;
        var dx = x2 - x1;
        return dy * dy + dx * dx;
    }
    function addVectors(a, b) {
        return { x: a.x + b.x, y: a.y + b.y };
    }

    var poseChain = [
        ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
        ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
        ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
        ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
        ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
        ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
        ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
        ['rightKnee', 'rightAnkle']
    ];
    var parentChildrenTuples = poseChain.map(function (_a) {
        var parentJoinName = _a[0], childJoinName = _a[1];
        return ([partIds[parentJoinName], partIds[childJoinName]]);
    });
    var parentToChildEdges = parentChildrenTuples.map(function (_a) {
        var childJointId = _a[1];
        return childJointId;
    });
    var childToParentEdges = parentChildrenTuples.map(function (_a) {
        var parentJointId = _a[0];
        return parentJointId;
    });
    function getDisplacement(i, point, displacements) {
        var numEdges = displacements.shape[2] / 2;
        return {
            y: displacements.get(point.y, point.x, i),
            x: displacements.get(point.y, point.x, numEdges + i)
        };
    }
    function decode(point, outputStride, height, width) {
        return {
            y: clamp(Math.round(point.y / outputStride), 0, height - 1),
            x: clamp(Math.round(point.x / outputStride), 0, width - 1)
        };
    }
    function traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scoresBuffer, offsets, outputStride, displacements) {
        var _a = scoresBuffer.shape, height = _a[0], width = _a[1];
        var sourceKeypointIndeces = decode(sourceKeypoint.position, outputStride, height, width);
        var displacement = getDisplacement(edgeId, sourceKeypointIndeces, displacements);
        var displacedPoint = addVectors(sourceKeypoint.position, displacement);
        var displacedPointIndeces = decode(displacedPoint, outputStride, height, width);
        var offsetPoint = getOffsetPoint(displacedPointIndeces.y, displacedPointIndeces.x, targetKeypointId, offsets);
        var targetKeypoint = addVectors(displacedPoint, { x: offsetPoint.x, y: offsetPoint.y });
        var targetKeypointIndeces = decode(targetKeypoint, outputStride, height, width);
        var score = scoresBuffer.get(targetKeypointIndeces.y, targetKeypointIndeces.x, targetKeypointId);
        return { position: targetKeypoint, part: partNames[targetKeypointId], score: score };
    }
    function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd) {
        var numParts = scores.shape[2];
        var numEdges = parentToChildEdges.length;
        var instanceKeypoints = new Array(numParts);
        var rootPart = root.part, rootScore = root.score;
        var rootPoint = getImageCoords(rootPart, outputStride, offsets);
        instanceKeypoints[rootPart.id] = {
            score: rootScore,
            part: partNames[rootPart.id],
            position: rootPoint
        };
        for (var edge = numEdges - 1; edge >= 0; --edge) {
            var sourceKeypointId = parentToChildEdges[edge];
            var targetKeypointId = childToParentEdges[edge];
            if (instanceKeypoints[sourceKeypointId] &&
                !instanceKeypoints[targetKeypointId]) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsBwd);
            }
        }
        for (var edge = 0; edge < numEdges; ++edge) {
            var sourceKeypointId = childToParentEdges[edge];
            var targetKeypointId = parentToChildEdges[edge];
            if (instanceKeypoints[sourceKeypointId] &&
                !instanceKeypoints[targetKeypointId]) {
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint(edge, instanceKeypoints[sourceKeypointId], targetKeypointId, scores, offsets, outputStride, displacementsFwd);
            }
        }
        return instanceKeypoints;
    }

    function withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, _a, keypointId) {
        var x = _a.x, y = _a.y;
        return poses.some(function (_a) {
            var keypoints = _a.keypoints;
            var correspondingKeypoint = keypoints[keypointId].position;
            return squaredDistance(y, x, correspondingKeypoint.y, correspondingKeypoint.x) <=
                squaredNmsRadius;
        });
    }
    function getInstanceScore(existingPoses, squaredNmsRadius, instanceKeypoints) {
        var notOverlappedKeypointScores = instanceKeypoints.reduce(function (result, _a, keypointId) {
            var position = _a.position, score = _a.score;
            if (!withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, position, keypointId)) {
                result += score;
            }
            return result;
        }, 0.0);
        return notOverlappedKeypointScores /= instanceKeypoints.length;
    }
    var kLocalMaximumRadius = 1;
    function decodeMultiplePoses(heatmapScores, offsets, displacementsFwd, displacementsBwd, outputStride, maxPoseDetections, scoreThreshold, nmsRadius) {
        if (scoreThreshold === void 0) { scoreThreshold = 0.5; }
        if (nmsRadius === void 0) { nmsRadius = 20; }
        return __awaiter(this, void 0, void 0, function () {
            var poses, _a, scoresBuffer, offsetsBuffer, displacementsFwdBuffer, displacementsBwdBuffer, queue, squaredNmsRadius, root, rootImageCoords, keypoints, score;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        poses = [];
                        return [4, toTensorBuffers3D([heatmapScores, offsets, displacementsFwd, displacementsBwd])];
                    case 1:
                        _a = _b.sent(), scoresBuffer = _a[0], offsetsBuffer = _a[1], displacementsFwdBuffer = _a[2], displacementsBwdBuffer = _a[3];
                        queue = buildPartWithScoreQueue(scoreThreshold, kLocalMaximumRadius, scoresBuffer);
                        squaredNmsRadius = nmsRadius * nmsRadius;
                        while (poses.length < maxPoseDetections && !queue.empty()) {
                            root = queue.dequeue();
                            rootImageCoords = getImageCoords(root.part, outputStride, offsetsBuffer);
                            if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootImageCoords, root.part.id)) {
                                continue;
                            }
                            keypoints = decodePose(root, scoresBuffer, offsetsBuffer, outputStride, displacementsFwdBuffer, displacementsBwdBuffer);
                            score = getInstanceScore(poses, squaredNmsRadius, keypoints);
                            poses.push({ keypoints: keypoints, score: score });
                        }
                        return [2, poses];
                }
            });
        });
    }

    var MANIFEST_FILE = 'manifest.json';
    var CheckpointLoader = (function () {
        function CheckpointLoader(urlPath) {
            this.urlPath = urlPath;
            if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
                this.urlPath += '/';
            }
        }
        CheckpointLoader.prototype.loadManifest = function () {
            var _this = this;
            return new Promise(function (resolve, reject) {
                var xhr = new XMLHttpRequest();
                xhr.open('GET', _this.urlPath + MANIFEST_FILE);
                xhr.onload = function () {
                    _this.checkpointManifest = JSON.parse(xhr.responseText);
                    resolve();
                };
                xhr.onerror = function (error) {
                    throw new Error(MANIFEST_FILE + " not found at " + _this.urlPath + ". " + error);
                };
                xhr.send();
            });
        };
        CheckpointLoader.prototype.getCheckpointManifest = function () {
            var _this = this;
            if (this.checkpointManifest == null) {
                return new Promise(function (resolve, reject) {
                    _this.loadManifest().then(function () {
                        resolve(_this.checkpointManifest);
                    });
                });
            }
            return new Promise(function (resolve, reject) {
                resolve(_this.checkpointManifest);
            });
        };
        CheckpointLoader.prototype.getAllVariables = function () {
            var _this = this;
            if (this.variables != null) {
                return new Promise(function (resolve, reject) {
                    resolve(_this.variables);
                });
            }
            return new Promise(function (resolve, reject) {
                _this.getCheckpointManifest().then(function (checkpointDefinition) {
                    var variableNames = Object.keys(_this.checkpointManifest);
                    var variablePromises = [];
                    for (var i = 0; i < variableNames.length; i++) {
                        variablePromises.push(_this.getVariable(variableNames[i]));
                    }
                    Promise.all(variablePromises).then(function (variables) {
                        _this.variables = {};
                        for (var i = 0; i < variables.length; i++) {
                            _this.variables[variableNames[i]] = variables[i];
                        }
                        resolve(_this.variables);
                    });
                });
            });
        };
        CheckpointLoader.prototype.getVariable = function (varName) {
            var _this = this;
            if (!(varName in this.checkpointManifest)) {
                throw new Error('Cannot load non-existant variable ' + varName);
            }
            var variableRequestPromiseMethod = function (resolve, reject) {
                var xhr = new XMLHttpRequest();
                xhr.responseType = 'arraybuffer';
                var fname = _this.checkpointManifest[varName].filename;
                xhr.open('GET', _this.urlPath + fname);
                xhr.onload = function () {
                    if (xhr.status === 404) {
                        throw new Error("Not found variable " + varName);
                    }
                    var values = new Float32Array(xhr.response);
                    var tensor = tf.Tensor.make(_this.checkpointManifest[varName].shape, { values: values });
                    resolve(tensor);
                };
                xhr.onerror = function (error) {
                    throw new Error("Could not fetch variable " + varName + ": " + error);
                };
                xhr.send();
            };
            if (this.checkpointManifest == null) {
                return new Promise(function (resolve, reject) {
                    _this.loadManifest().then(function () {
                        new Promise(variableRequestPromiseMethod).then(resolve);
                    });
                });
            }
            return new Promise(variableRequestPromiseMethod);
        };
        return CheckpointLoader;
    }());

    var mobileNet100Architecture = [
        ['conv2d', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1]
    ];
    var mobileNet75Architecture = [
        ['conv2d', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1]
    ];
    var mobileNet50Architecture = [
        ['conv2d', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 2],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1],
        ['separableConv', 1]
    ];
    var VALID_OUTPUT_STRIDES = [8, 16, 32];
    function assertValidOutputStride(outputStride) {
        tf.util.assert(typeof outputStride === 'number', 'outputStride is not a number');
        tf.util.assert(VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0, "outputStride of " + outputStride + " is invalid. " +
            "It must be either 8, 16, or 32");
    }
    function assertValidScaleFactor(imageScaleFactor) {
        tf.util.assert(typeof imageScaleFactor === 'number', 'imageScaleFactor is not a number');
        tf.util.assert(imageScaleFactor >= 0.2 && imageScaleFactor <= 1.0, 'imageScaleFactor must be between 0.2 and 1.0');
    }
    var mobileNetArchitectures = {
        100: mobileNet100Architecture,
        75: mobileNet75Architecture,
        50: mobileNet50Architecture
    };
    function toOutputStridedLayers(convolutionDefinition, outputStride) {
        var currentStride = 1;
        var rate = 1;
        return convolutionDefinition.map(function (_a, blockId) {
            var convType = _a[0], stride = _a[1];
            var layerStride, layerRate;
            if (currentStride === outputStride) {
                layerStride = 1;
                layerRate = rate;
                rate *= stride;
            }
            else {
                layerStride = stride;
                layerRate = 1;
                currentStride *= stride;
            }
            return {
                blockId: blockId, convType: convType, stride: layerStride, rate: layerRate,
                outputStride: currentStride
            };
        });
    }
    var MobileNet = (function () {
        function MobileNet(variables, convolutionDefinitions) {
            this.PREPROCESS_DIVISOR = tf.scalar(255.0 / 2);
            this.ONE = tf.scalar(1);
            this.variables = variables;
            this.convolutionDefinitions = convolutionDefinitions;
        }
        MobileNet.prototype.predict = function (input, outputStride) {
            var _this = this;
            var preprocessedInput = tf.cast(input, 'float32').div(this.PREPROCESS_DIVISOR).sub(this.ONE);
            var layers = toOutputStridedLayers(this.convolutionDefinitions, outputStride);
            return layers.reduce(function (previousLayer, _a) {
                var blockId = _a.blockId, stride = _a.stride, convType = _a.convType, rate = _a.rate;
                if (convType === 'conv2d') {
                    return _this.conv(previousLayer, stride, blockId);
                }
                else if (convType === 'separableConv') {
                    return _this.separableConv(previousLayer, stride, blockId, rate);
                }
                else {
                    throw Error('Unknown conv type of ' + convType);
                }
            }, preprocessedInput);
        };
        MobileNet.prototype.convToOutput = function (mobileNetOutput, outputLayerName) {
            return mobileNetOutput.conv2d(this.weights(outputLayerName), 1, 'same')
                .add(this.biases(outputLayerName));
        };
        MobileNet.prototype.conv = function (inputs, stride, blockId) {
            return inputs
                .conv2d(this.weights("Conv2d_" + String(blockId)), stride, 'same')
                .add(this.biases("Conv2d_" + String(blockId)))
                .clipByValue(0, 6);
        };
        MobileNet.prototype.separableConv = function (inputs, stride, blockID, dilations) {
            if (dilations === void 0) { dilations = 1; }
            var dwLayer = "Conv2d_" + String(blockID) + "_depthwise";
            var pwLayer = "Conv2d_" + String(blockID) + "_pointwise";
            var x1 = inputs
                .depthwiseConv2D(this.depthwiseWeights(dwLayer), stride, 'same', 'NHWC', dilations)
                .add(this.biases(dwLayer))
                .clipByValue(0, 6);
            var x2 = x1.conv2d(this.weights(pwLayer), [1, 1], 'same')
                .add(this.biases(pwLayer))
                .clipByValue(0, 6);
            return x2;
        };
        MobileNet.prototype.weights = function (layerName) {
            return this.variables["MobilenetV1/" + layerName + "/weights"];
        };
        MobileNet.prototype.biases = function (layerName) {
            return this.variables["MobilenetV1/" + layerName + "/biases"];
        };
        MobileNet.prototype.depthwiseWeights = function (layerName) {
            return this.variables["MobilenetV1/" + layerName + "/depthwise_weights"];
        };
        MobileNet.prototype.dispose = function () {
            for (var varName in this.variables) {
                this.variables[varName].dispose();
            }
        };
        return MobileNet;
    }());

    var GOOGLE_CLOUD_STORAGE_DIR = 'https://storage.googleapis.com/tfjs-models/weights/posenet/';
    var checkpoints = {
        1.01: {
            url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_101/',
            architecture: mobileNetArchitectures[100]
        },
        1.0: {
            url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_100/',
            architecture: mobileNetArchitectures[100]
        },
        0.75: {
            url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_075/',
            architecture: mobileNetArchitectures[75]
        },
        0.5: {
            url: GOOGLE_CLOUD_STORAGE_DIR + 'mobilenet_v1_050/',
            architecture: mobileNetArchitectures[50]
        }
    };

    function mod(a, b) {
        return tf.tidy(function () {
            var floored = a.div(tf.scalar(b, 'int32'));
            return a.sub(floored.mul(tf.scalar(b, 'int32')));
        });
    }
    function argmax2d(inputs) {
        var _a = inputs.shape, height = _a[0], width = _a[1], depth = _a[2];
        return tf.tidy(function () {
            var reshaped = inputs.reshape([height * width, depth]);
            var coords = reshaped.argMax(0);
            var yCoords = coords.div(tf.scalar(width, 'int32')).expandDims(1);
            var xCoords = mod(coords, width).expandDims(1);
            return tf.concat([yCoords, xCoords], 1);
        });
    }

    function getPointsConfidence(heatmapScores, heatMapCoords) {
        var numKeypoints = heatMapCoords.shape[0];
        var result = new Float32Array(numKeypoints);
        for (var keypoint = 0; keypoint < numKeypoints; keypoint++) {
            var y = heatMapCoords.get(keypoint, 0);
            var x = heatMapCoords.get(keypoint, 1);
            result[keypoint] = heatmapScores.get(y, x, keypoint);
        }
        return result;
    }
    function getOffsetPoint$1(y, x, keypoint, offsetsBuffer) {
        return {
            y: offsetsBuffer.get(y, x, keypoint),
            x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS)
        };
    }
    function getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer) {
        var result = [];
        for (var keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
            var heatmapY = heatMapCoordsBuffer.get(keypoint, 0).valueOf();
            var heatmapX = heatMapCoordsBuffer.get(keypoint, 1).valueOf();
            var _a = getOffsetPoint$1(heatmapY, heatmapX, keypoint, offsetsBuffer), x = _a.x, y = _a.y;
            result.push(y);
            result.push(x);
        }
        return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
    }
    function getOffsetPoints(heatMapCoordsBuffer, outputStride, offsetsBuffer) {
        return tf.tidy(function () {
            var offsetVectors = getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer);
            return heatMapCoordsBuffer.toTensor()
                .mul(tf.scalar(outputStride, 'int32'))
                .toFloat()
                .add(offsetVectors);
        });
    }

    function decodeSinglePose(heatmapScores, offsets, outputStride) {
        return __awaiter(this, void 0, void 0, function () {
            var totalScore, heatmapValues, _a, scoresBuffer, offsetsBuffer, heatmapValuesBuffer, offsetPoints, offsetPointsBuffer, keypointConfidence, keypoints;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        totalScore = 0.0;
                        heatmapValues = argmax2d(heatmapScores);
                        return [4, Promise.all([
                                toTensorBuffer(heatmapScores), toTensorBuffer(offsets),
                                toTensorBuffer(heatmapValues, 'int32')
                            ])];
                    case 1:
                        _a = _b.sent(), scoresBuffer = _a[0], offsetsBuffer = _a[1], heatmapValuesBuffer = _a[2];
                        offsetPoints = getOffsetPoints(heatmapValuesBuffer, outputStride, offsetsBuffer);
                        return [4, toTensorBuffer(offsetPoints)];
                    case 2:
                        offsetPointsBuffer = _b.sent();
                        keypointConfidence = Array.from(getPointsConfidence(scoresBuffer, heatmapValuesBuffer));
                        keypoints = keypointConfidence.map(function (score, keypointId) {
                            totalScore += score;
                            return {
                                position: {
                                    y: offsetPointsBuffer.get(keypointId, 0),
                                    x: offsetPointsBuffer.get(keypointId, 1)
                                },
                                part: partNames[keypointId],
                                score: score
                            };
                        });
                        heatmapValues.dispose();
                        offsetPoints.dispose();
                        return [2, { keypoints: keypoints, score: totalScore / keypoints.length }];
                }
            });
        });
    }

    function toInputTensor(input, resizeHeight, resizeWidth, flipHorizontal) {
        var imageTensor = tf.fromPixels(input);
        if (flipHorizontal) {
            return imageTensor.reverse(1).resizeBilinear([resizeHeight, resizeWidth]);
        }
        else {
            return imageTensor.resizeBilinear([resizeHeight, resizeWidth]);
        }
    }
    var PoseNet = (function () {
        function PoseNet(mobileNet) {
            this.mobileNet = mobileNet;
        }
        PoseNet.prototype.predictForSinglePose = function (input, outputStride) {
            var _this = this;
            if (outputStride === void 0) { outputStride = 16; }
            assertValidOutputStride(outputStride);
            return tf.tidy(function () {
                var mobileNetOutput = _this.mobileNet.predict(input, outputStride);
                var heatmaps = _this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');
                var offsets = _this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');
                return { heatmapScores: heatmaps.sigmoid(), offsets: offsets };
            });
        };
        PoseNet.prototype.predictForMultiPose = function (input, outputStride) {
            var _this = this;
            if (outputStride === void 0) { outputStride = 16; }
            return tf.tidy(function () {
                var mobileNetOutput = _this.mobileNet.predict(input, outputStride);
                var heatmaps = _this.mobileNet.convToOutput(mobileNetOutput, 'heatmap_2');
                var offsets = _this.mobileNet.convToOutput(mobileNetOutput, 'offset_2');
                var displacementFwd = _this.mobileNet.convToOutput(mobileNetOutput, 'displacement_fwd_2');
                var displacementBwd = _this.mobileNet.convToOutput(mobileNetOutput, 'displacement_bwd_2');
                return {
                    heatmapScores: heatmaps.sigmoid(),
                    offsets: offsets,
                    displacementFwd: displacementFwd,
                    displacementBwd: displacementBwd
                };
            });
        };
        PoseNet.prototype.estimateSinglePose = function (input, imageScaleFactor, flipHorizontal, outputStride) {
            if (imageScaleFactor === void 0) { imageScaleFactor = 0.5; }
            if (flipHorizontal === void 0) { flipHorizontal = false; }
            if (outputStride === void 0) { outputStride = 16; }
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                var resizedHeight, resizedWidth, _a, heatmapScores, offsets, pose, scaleY, scaleX;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            assertValidOutputStride(outputStride);
                            assertValidScaleFactor(imageScaleFactor);
                            resizedHeight = getValidResolution(imageScaleFactor, input.height, outputStride);
                            resizedWidth = getValidResolution(imageScaleFactor, input.width, outputStride);
                            _a = tf.tidy(function () {
                                var inputTensor = toInputTensor(input, resizedHeight, resizedWidth, flipHorizontal);
                                return _this.predictForSinglePose(inputTensor, outputStride);
                            }), heatmapScores = _a.heatmapScores, offsets = _a.offsets;
                            return [4, decodeSinglePose(heatmapScores, offsets, outputStride)];
                        case 1:
                            pose = _b.sent();
                            heatmapScores.dispose();
                            offsets.dispose();
                            scaleY = input.height / resizedHeight;
                            scaleX = input.width / resizedWidth;
                            return [2, scalePose(pose, scaleY, scaleX)];
                    }
                });
            });
        };
        PoseNet.prototype.estimateMultiplePoses = function (input, imageScaleFactor, flipHorizontal, outputStride, maxDetections, scoreThreshold, nmsRadius) {
            if (imageScaleFactor === void 0) { imageScaleFactor = 0.5; }
            if (flipHorizontal === void 0) { flipHorizontal = false; }
            if (outputStride === void 0) { outputStride = 16; }
            if (maxDetections === void 0) { maxDetections = 5; }
            if (scoreThreshold === void 0) { scoreThreshold = .5; }
            if (nmsRadius === void 0) { nmsRadius = 20; }
            return __awaiter(this, void 0, void 0, function () {
                var _this = this;
                var resizedHeight, resizedWidth, _a, heatmapScores, offsets, displacementFwd, displacementBwd, poses, scaleY, scaleX;
                return __generator(this, function (_b) {
                    switch (_b.label) {
                        case 0:
                            assertValidOutputStride(outputStride);
                            assertValidScaleFactor(imageScaleFactor);
                            resizedHeight = getValidResolution(imageScaleFactor, input.height, outputStride);
                            resizedWidth = getValidResolution(imageScaleFactor, input.width, outputStride);
                            _a = tf.tidy(function () {
                                var inputTensor = toInputTensor(input, resizedHeight, resizedWidth, flipHorizontal);
                                return _this.predictForMultiPose(inputTensor, outputStride);
                            }), heatmapScores = _a.heatmapScores, offsets = _a.offsets, displacementFwd = _a.displacementFwd, displacementBwd = _a.displacementBwd;
                            return [4, decodeMultiplePoses(heatmapScores, offsets, displacementFwd, displacementBwd, outputStride, maxDetections, scoreThreshold, nmsRadius)];
                        case 1:
                            poses = _b.sent();
                            heatmapScores.dispose();
                            offsets.dispose();
                            displacementFwd.dispose();
                            displacementBwd.dispose();
                            scaleY = input.height / resizedHeight;
                            scaleX = input.width / resizedWidth;
                            return [2, scalePoses(poses, scaleY, scaleX)];
                    }
                });
            });
        };
        PoseNet.prototype.dispose = function () {
            this.mobileNet.dispose();
        };
        return PoseNet;
    }());
    function load(multiplier) {
        if (multiplier === void 0) { multiplier = 1.01; }
        return __awaiter(this, void 0, void 0, function () {
            var possibleMultipliers, checkpoint, checkpointLoader, variables, mobileNet;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (tf == null) {
                            throw new Error("Cannot find TensorFlow.js. If you are using a <script> tag, please " +
                                "also include @tensorflow/tfjs on the page before using this model.");
                        }
                        possibleMultipliers = Object.keys(checkpoints);
                        tf.util.assert(typeof multiplier === 'number', "got multiplier type of " + typeof multiplier + " when it should be a " +
                            "number.");
                        tf.util.assert(possibleMultipliers.indexOf(multiplier.toString()) >= 0, "invalid multiplier value of " + multiplier + ".  No checkpoint exists for that " +
                            ("multiplier. Must be one of " + possibleMultipliers.join(',') + "."));
                        checkpoint = checkpoints[multiplier];
                        checkpointLoader = new CheckpointLoader(checkpoint.url);
                        return [4, checkpointLoader.getAllVariables()];
                    case 1:
                        variables = _a.sent();
                        mobileNet = new MobileNet(variables, checkpoint.architecture);
                        return [2, new PoseNet(mobileNet)];
                }
            });
        });
    }

    exports.decodeMultiplePoses = decodeMultiplePoses;
    exports.decodeSinglePose = decodeSinglePose;
    exports.load = load;
    exports.PoseNet = PoseNet;
    exports.checkpoints = checkpoints;
    exports.partIds = partIds;
    exports.partNames = partNames;
    exports.getAdjacentKeyPoints = getAdjacentKeyPoints;
    exports.getBoundingBox = getBoundingBox;
    exports.getBoundingBoxPoints = getBoundingBoxPoints;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
