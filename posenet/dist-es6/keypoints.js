export var partNames = [
    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
];
export var NUM_KEYPOINTS = partNames.length;
export var partIds = partNames.reduce(function (result, jointName, i) {
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
export var connectedPartIndeces = connectedPartNames.map(function (_a) {
    var jointNameA = _a[0], jointNameB = _a[1];
    return ([partIds[jointNameA], partIds[jointNameB]]);
});
//# sourceMappingURL=keypoints.js.map