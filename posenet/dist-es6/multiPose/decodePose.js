import { partIds, partNames } from '../keypoints';
import { clamp, getOffsetPoint } from './util';
import { addVectors, getImageCoords } from './util';
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
export function decodePose(root, scores, offsets, outputStride, displacementsFwd, displacementsBwd) {
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
//# sourceMappingURL=decodePose.js.map