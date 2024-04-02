
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export interface ObjectDetectionClass {
  name: string;
  id: number;
  displayName: string;
}

export const CLASSES: {[key: string]: ObjectDetectionClass} = {
  1: {
    name: '/m/01g317',
    id: 1,
    displayName: 'plastic bottle',
  },
  2: {
    name: '/m/0199g',
    id: 2,
    displayName: 'bottle',
  },
  3: {
    name: '/m/0k4j',
    id: 3,
    displayName: 'plastic bags',
  },
  4: {
    name: '/m/04_sv',
    id: 4,
    displayName: 'paper plates',
  },
  5: {
    name: '/m/05czz6l',
    id: 5,
    displayName: 'paper bags',
  },
  6: {
    name: '/m/01bjv',
    id: 6,
    displayName: 'aluminium can',
  },
  7: {
    name: '/m/07jdr',
    id: 7,
    displayName: 'paper cup',
  },
  8: {
    name: '/m/07r04',
    id: 8,
    displayName: 'crushed can',
  },
  9: {
    name: '/m/019jd',
    id: 9,
    displayName: 'aluminium can',
  },
  10: {
    name: '/m/015qff',
    id: 10,
    displayName: 'packet juice',
  },
  11: {
    name: '/m/01pns0',
    id: 11,
    displayName: 'glass pieces',
  },
  13: {
    name: '/m/02pv19',
    id: 13,
    displayName: 'cigarette butts',
  },
  14: {
    name: '/m/015qbp',
    id: 14,
    displayName: 'crushed bottle',
  },
  15: {
    name: '/m/0cvnqh',
    id: 15,
    displayName: 'nets',
  },
  16: {
    name: '/m/015p6',
    id: 16,
    displayName: 'fishing nets',
  },
  17: {
    name: '/m/01yrx',
    id: 17,
    displayName: 'cat',
  },
  18: {
    name: '/m/0bt9lr',
    id: 18,
    displayName: 'dog',
  },
  19: {
    name: '/m/03k3r',
    id: 19,
    displayName: 'person',
  },
  20: {
    name: '/m/07bgp',
    id: 20,
    displayName: 'cloth',
  },
  21: {
    name: '/m/01xq0k1',
    id: 21,
    displayName: 'food waste',
  },
  22: {
    name: '/m/0bwd_0j',
    id: 22,
    displayName: 'elephant',
  },
  23: {
    name: '/m/01dws',
    id: 23,
    displayName: 'bear',
  },
  24: {
    name: '/m/0898b',
    id: 24,
    displayName: 'zebra',
  },
  25: {
    name: '/m/03bk1',
    id: 25,
    displayName: 'giraffe',
  },
  27: {
    name: '/m/01940j',
    id: 27,
    displayName: 'backpack',
  },
  28: {
    name: '/m/0hnnb',
    id: 28,
    displayName: 'umbrella',
  },
  31: {
    name: '/m/080hkjn',
    id: 31,
    displayName: 'handbag',
  },
  32: {
    name: '/m/01rkbr',
    id: 32,
    displayName: 'tie',
  },
  33: {
    name: '/m/01s55n',
    id: 33,
    displayName: 'suitcase',
  },
  34: {
    name: '/m/02wmf',
    id: 34,
    displayName: 'frisbee',
  },
  35: {
    name: '/m/06__v',
    id: 36,
    displayName: 'surfboard',
  },
  36: {
    name: '/m/018xm',
    id: 37,
    displayName: 'sports ball',
  },
  37: {
    name: '/m/02zt3',
    id: 38,
    displayName: 'kite',
  },
  38: {
    name: '/m/03g8mr',
    id: 39,
    displayName: 'baseball bat',
  },
  39: {
    name: '/m/03grzl',
    id: 40,
    displayName: 'baseball glove',
  },
  40: {
    name: '/m/06_fw',
    id: 41,
    displayName: 'skateboard',
  },
  41: {
    name: '/m/09tvcd',
    id: 46,
    displayName: 'wine glass',
  },
  42: {
    name: '/m/08gqpm',
    id: 47,
    displayName: 'cup',
  },
  43: {
    name: '/m/0dt3t',
    id: 48,
    displayName: 'fork',
  },
  44: {
    name: '/m/04ctx',
    id: 49,
    displayName: 'knife',
  },
  45: {
    name: '/m/0cmx8',
    id: 50,
    displayName: 'spoon',
  },
  46: {
    name: '/m/04kkgm',
    id: 51,
    displayName: 'bowl',
  },
  47: {
    name: '/m/09qck',
    id: 52,
    displayName: 'banana',
  },
  48: {
    name: '/m/014j1m',
    id: 53,
    displayName: 'apple',
  },
  49: {
    name: '/m/0l515',
    id: 54,
    displayName: 'sandwich',
  },
  50: {
    name: '/m/0cyhj_',
    id: 55,
    displayName: 'orange',
  },
  51: {
    name: '/m/0hkxq',
    id: 56,
    displayName: 'broccoli',
  },
  52: {
    name: '/m/0fj52s',
    id: 57,
    displayName: 'carrot',
  },
  53: {
    name: '/m/01b9xk',
    id: 58,
    displayName: 'ice cream cup',
  },
  54: {
    name: '/m/0663v',
    id: 59,
    displayName: 'paper',
  },
  55: {
    name: '/m/0jy4k',
    id: 60,
    displayName: 'donut',
  },
  56: {
    name: '/m/0fszt',
    id: 61,
    displayName: 'cardboard',
  },
  57: {
    name: '/m/0bt_c3',
    id: 84,
    displayName: 'book',
  },
  58: {
    name: '/m/012xff',
    id: 90,
    displayName: 'toothbrush',
  }
};
