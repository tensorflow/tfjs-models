import { NUM_KEYPOINTS } from '../keypoints';
export function getOffsetPoint(y, x, keypoint, offsets) {
    return {
        y: offsets.get(y, x, keypoint),
        x: offsets.get(y, x, keypoint + NUM_KEYPOINTS)
    };
}
export function getImageCoords(part, outputStride, offsets) {
    var heatmapY = part.heatmapY, heatmapX = part.heatmapX, keypoint = part.id;
    var _a = getOffsetPoint(heatmapY, heatmapX, keypoint, offsets), y = _a.y, x = _a.x;
    return {
        x: part.heatmapX * outputStride + x,
        y: part.heatmapY * outputStride + y
    };
}
export function fillArray(element, size) {
    var result = new Array(size);
    for (var i = 0; i < size; i++) {
        result[i] = element;
    }
    return result;
}
export function clamp(a, min, max) {
    if (a < min) {
        return min;
    }
    if (a > max) {
        return max;
    }
    return a;
}
export function squaredDistance(y1, x1, y2, x2) {
    var dy = y2 - y1;
    var dx = x2 - x1;
    return dy * dy + dx * dx;
}
export function addVectors(a, b) {
    return { x: a.x + b.x, y: a.y + b.y };
}
export function clampVector(a, min, max) {
    return { y: clamp(a.y, min, max), x: clamp(a.x, min, max) };
}
//# sourceMappingURL=util.js.map