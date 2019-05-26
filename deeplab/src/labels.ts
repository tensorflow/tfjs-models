import { Color, Label } from './model';

const DATASET_MAX_ENTRIES = {
    PASCAL: 256,
};

const createPascalColormap = () => {
    const colormap = new Array(DATASET_MAX_ENTRIES.PASCAL);
    for (let idx = 0; idx < DATASET_MAX_ENTRIES.PASCAL; ++idx) {
        colormap[idx] = new Array(3);
    }
    for (let shift = 7; shift > 4; --shift) {
        const indexShift = 3 * (7 - shift);
        for (let channel = 0; channel < 3; ++channel) {
            for (let idx = 0; idx < DATASET_MAX_ENTRIES.PASCAL; ++idx) {
                colormap[idx][channel] |=
                    ((idx >> (channel + indexShift)) & 1) << shift;
            }
        }
    }
    return colormap;
};

export const COLORMAP: Color[] = createPascalColormap();
export const LABELS: Label[] = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'TV',
];

// for shift in reversed(range(8)):
// for channel in range(3):
//   colormap[:, channel] |= ((ind >> channel) & 1) << shift
// ind >>= 3
