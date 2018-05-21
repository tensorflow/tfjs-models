import { mobileNetArchitectures } from './mobilenet';
var GOOGLE_CLOUD_STORAGE_DIR = 'https://storage.googleapis.com/tfjs-models/weights/posenet/';
export var checkpoints = {
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
//# sourceMappingURL=checkpoints.js.map