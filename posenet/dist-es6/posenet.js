var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
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
};
import * as tf from '@tensorflow/tfjs';
import { CheckpointLoader } from './checkpoint_loader';
import { checkpoints } from './checkpoints';
import { assertValidOutputStride, assertValidScaleFactor, MobileNet } from './mobilenet';
import decodeMultiplePoses from './multiPose/decodeMultiplePoses';
import decodeSinglePose from './singlePose/decodeSinglePose';
import { getValidResolution, scalePose, scalePoses } from './util';
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
export { PoseNet };
export function load(multiplier) {
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
//# sourceMappingURL=posenet.js.map