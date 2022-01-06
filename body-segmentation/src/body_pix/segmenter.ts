/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import {BodySegmenter} from '../body_segmenter';
import {Mask, Segmentation} from '../shared/calculators/interfaces/common_interfaces';
import {assertMaskValue, toHTMLCanvasElementLossy, toTensorLossy} from '../shared/calculators/mask_util';
import {BodySegmenterInput} from '../types';

import * as bodyPix from './impl';
import {BodyPixModelConfig, BodyPixSegmentationConfig} from './types';

class BodyPixMask implements Mask {
  constructor(private mask: ImageData) {}

  async toCanvasImageSource() {
    return toHTMLCanvasElementLossy(this.mask);
  }

  async toImageData() {
    return this.mask;
  }

  async toTensor() {
    return toTensorLossy(this.mask);
  }

  getUnderlyingType() {
    return 'imagedata' as const ;
  }
}

function singleMaskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  if (maskValue !== 255) {
    throw new Error(`Foreground id must be 255 but got ${maskValue}`);
  }
  return 'person';
}

function multiMaskValueToLabel(maskValue: number) {
  assertMaskValue(maskValue);
  if (maskValue >= bodyPix.PART_CHANNELS.length) {
    throw new Error(`Invalid body part value ${maskValue}`);
  }
  return bodyPix.PART_CHANNELS[maskValue];
}

/**
 * MediaPipe segmenter class.
 */
class BodyPixSegmenter implements BodySegmenter {
  private readonly bodyPixModel: bodyPix.BodyPix;

  // Should not be called outside.
  constructor(model: bodyPix.BodyPix) {
    this.bodyPixModel = model;
  }

  /**
   * Segment people found in an image or video frame.
   *
   * It returns a single segmentation which contains all the detected people
   * in the input.
   *
   * @param input
   * ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement The input
   * image to feed through the network.
   *
   * @param config Optional.
   *       flipHorizontal: Optional. Default to false. When image data comes
   *       from camera, the result has to flip horizontally.
   *
   * @return An array of one `Segmentation`.
   */
  async segmentPeople(
      input: BodySegmenterInput,
      segmentationConfig: BodyPixSegmentationConfig): Promise<Segmentation[]> {
    if (input instanceof ImageBitmap) {
      const canvas = document.createElement('canvas');
      canvas.getContext('2d').drawImage(input, 0, 0);
      input = canvas;
    }

    let segmentations: Segmentation[];

    if (segmentationConfig.segmentBodyParts) {
      type PartSegmentation = {data: Int32Array, width: number, height: number};

      const partSegmentations: PartSegmentation[] =
          segmentationConfig.multiSegmentation ?
          await this.bodyPixModel.segmentMultiPersonParts(
              input, segmentationConfig) :
          [await this.bodyPixModel.segmentPersonParts(
              input, segmentationConfig)];

      segmentations = partSegmentations.map(partSegmentation => {
        const {data, width, height} = partSegmentation;
        const rgbaData = new Uint8ClampedArray(width * height * 4).fill(0);
        data.forEach((bodyPartLabel, i) => {
          // Background.
          if (bodyPartLabel === -1) {
            rgbaData[i * 4] = bodyPix.PART_CHANNELS.length;
            rgbaData[i * 4 + 3] = 0;
          } else {
            rgbaData[i * 4] = bodyPartLabel;
            rgbaData[i * 4 + 3] = 255;
          }
        });

        return {
          maskValueToLabel: multiMaskValueToLabel,
          mask: new BodyPixMask(new ImageData(rgbaData, width, height)),
        };
      });
    } else {
      type SingleSegmentation = {
        data: Uint8Array,
        width: number,
        height: number
      };

      const singleSegmentations: SingleSegmentation[] =
          segmentationConfig.multiSegmentation ?
          await this.bodyPixModel.segmentMultiPerson(
              input, segmentationConfig) :
          [await this.bodyPixModel.segmentPerson(input, segmentationConfig)];

      segmentations = singleSegmentations.map(singleSegmentation => {
        const {data, width, height} = singleSegmentation;
        const rgbaData = new Uint8ClampedArray(width * height * 4).fill(0);
        data.forEach((bodyPartLabel, i) => {
          // Background.
          if (bodyPartLabel === 0) {
            rgbaData[i * 4] = 0;
            rgbaData[i * 4 + 3] = 0;
          } else {
            rgbaData[i * 4] = 255;
            rgbaData[i * 4 + 3] = 255;
          }
        });

        return {
          maskValueToLabel: singleMaskValueToLabel,
          mask: new BodyPixMask(new ImageData(rgbaData, width, height)),
        };
      });
    }

    return segmentations;
  }

  dispose() {
    this.bodyPixModel.dispose();
  }

  reset() {}
}

/**
 * Loads the BodyPix solution.
 *
 * @param modelConfig An object that contains parameters for
 * the BodyPix loading process. Please find more details of
 * each parameters in the documentation of the
 * `BodyPixModelConfig` interface.
 */
export async function load(modelConfig?: BodyPixModelConfig):
    Promise<BodySegmenter> {
  return bodyPix.load(modelConfig).then(model => new BodyPixSegmenter(model));
}
