import * as tf from '@tensorflow/tfjs';
import { NUM_KEYPOINTS } from '../keypoints';
export function getPointsConfidence(heatmapScores, heatMapCoords) {
    var numKeypoints = heatMapCoords.shape[0];
    var result = new Float32Array(numKeypoints);
    for (var keypoint = 0; keypoint < numKeypoints; keypoint++) {
        var y = heatMapCoords.get(keypoint, 0);
        var x = heatMapCoords.get(keypoint, 1);
        result[keypoint] = heatmapScores.get(y, x, keypoint);
    }
    return result;
}
function getOffsetPoint(y, x, keypoint, offsetsBuffer) {
    return {
        y: offsetsBuffer.get(y, x, keypoint),
        x: offsetsBuffer.get(y, x, keypoint + NUM_KEYPOINTS)
    };
}
export function getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer) {
    var result = [];
    for (var keypoint = 0; keypoint < NUM_KEYPOINTS; keypoint++) {
        var heatmapY = heatMapCoordsBuffer.get(keypoint, 0).valueOf();
        var heatmapX = heatMapCoordsBuffer.get(keypoint, 1).valueOf();
        var _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsetsBuffer), x = _a.x, y = _a.y;
        result.push(y);
        result.push(x);
    }
    return tf.tensor2d(result, [NUM_KEYPOINTS, 2]);
}
export function getOffsetPoints(heatMapCoordsBuffer, outputStride, offsetsBuffer) {
    return tf.tidy(function () {
        var offsetVectors = getOffsetVectors(heatMapCoordsBuffer, offsetsBuffer);
        return heatMapCoordsBuffer.toTensor()
            .mul(tf.scalar(outputStride, 'int32'))
            .toFloat()
            .add(offsetVectors);
    });
}
//# sourceMappingURL=util.js.map