
const GOOGLE_CLOUD_STORAGE_DIR =
    'https://storage.googleapis.com/cl-move-mirror.appspot.com/';

import {mobileNetArchitectures, ConvolutionDefinition} from './mobilenet'

// https://storage.cloud.google.com/cl-move-mirror.appspot.com/mobile_net_101/manifest.json?_ga=2.16958649.-1265528061.1521221796

export type Checkpoint = {
  url: string,
  architecture: ConvolutionDefinition[]
}

export const checkpoints: {[name: string]: Checkpoint} = {
  '101': {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobile_net_101/',
    architecture: mobileNetArchitectures[100]
  },
  '100': {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobile_net_100/',
    architecture: mobileNetArchitectures[100]
  },
  '75': {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobile_net_075/',
    architecture: mobileNetArchitectures[75]
  },
  '50': {
    url: GOOGLE_CLOUD_STORAGE_DIR + 'mobile_net_050/',
    architecture: mobileNetArchitectures[50]
  }
}
