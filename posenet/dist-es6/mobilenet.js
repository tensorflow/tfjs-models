import * as tf from '@tensorflow/tfjs';
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
export function assertValidOutputStride(outputStride) {
    tf.util.assert(typeof outputStride === 'number', 'outputStride is not a number');
    tf.util.assert(VALID_OUTPUT_STRIDES.indexOf(outputStride) >= 0, "outputStride of " + outputStride + " is invalid. " +
        "It must be either 8, 16, or 32");
}
export function assertValidResolution(resolution, outputStride) {
    tf.util.assert(typeof resolution === 'number', 'resolution is not a number');
    tf.util.assert((resolution - 1) % outputStride === 0, "resolution of " + resolution + " is invalid for output stride " +
        (outputStride + "."));
}
export function assertValidScaleFactor(imageScaleFactor) {
    tf.util.assert(typeof imageScaleFactor === 'number', 'imageScaleFactor is not a number');
    tf.util.assert(imageScaleFactor >= 0.2 && imageScaleFactor <= 1.0, 'imageScaleFactor must be between 0.2 and 1.0');
}
export var mobileNetArchitectures = {
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
export { MobileNet };
//# sourceMappingURL=mobilenet.js.map