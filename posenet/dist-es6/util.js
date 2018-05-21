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
import { connectedPartIndeces } from './keypoints';
function eitherPointDoesntMeetConfidence(a, b, minConfidence) {
    return (a < minConfidence || b < minConfidence);
}
export function getAdjacentKeyPoints(keypoints, minConfidence) {
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
export function getBoundingBox(keypoints) {
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
export function getBoundingBoxPoints(keypoints) {
    var _a = getBoundingBox(keypoints), minX = _a.minX, minY = _a.minY, maxX = _a.maxX, maxY = _a.maxY;
    return [
        { x: minX, y: minY }, { x: maxX, y: minY }, { x: maxX, y: maxY },
        { x: minX, y: maxY }
    ];
}
export function toTensorBuffer(tensor, type) {
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
export function toTensorBuffers3D(tensors) {
    return __awaiter(this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            return [2, Promise.all(tensors.map(function (tensor) { return toTensorBuffer(tensor, 'float32'); }))];
        });
    });
}
export function scalePose(pose, scaleX, scaleY) {
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
export function scalePoses(poses, scaleY, scaleX) {
    if (scaleX === 1 && scaleY === 1) {
        return poses;
    }
    return poses.map(function (pose) { return scalePose(pose, scaleX, scaleY); });
}
export function getValidResolution(imageScaleFactor, inputDimension, outputStride) {
    var evenResolution = inputDimension * imageScaleFactor - 1;
    return evenResolution - (evenResolution % outputStride) + 1;
}
//# sourceMappingURL=util.js.map