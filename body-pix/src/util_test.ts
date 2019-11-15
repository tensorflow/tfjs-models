/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
 *
 * =============================================================================
 */

// tslint:disable-next-line: no-imports-from-dist
import {ALL_ENVS, describeWithFlags} from '@tensorflow/tfjs-core/dist/jasmine_util';

import {BodyPixOutputStride} from './types';
import {toInputResolutionHeightAndWidth, toValidInputResolution} from './util';

describeWithFlags('util.toValidInputResolution', ALL_ENVS, () => {
  it('returns an odd value', () => {
    expect(toValidInputResolution(1920, 8) % 2).toEqual(1);
    expect(toValidInputResolution(1280, 16) % 2).toEqual(1);
    expect(toValidInputResolution(719, 16) % 2).toEqual(1);
    expect(toValidInputResolution(545, 16) % 2).toEqual(1);
    expect(toValidInputResolution(225, 8) % 2).toEqual(1);
    expect(toValidInputResolution(240, 8) % 2).toEqual(1);
  });

  it('returns the original value when already a valid resolution', () => {
    const outputStride = 16;

    const validResolution = toValidInputResolution(1000, outputStride);

    const resolution = toValidInputResolution(validResolution, outputStride);

    expect(resolution).toEqual(validResolution);
  });

  it('succeeds when 1-resolution is divisible by the output stride', () => {
    const outputStride = 8;
    const inputResolution = 562;

    const resolution = toValidInputResolution(inputResolution, outputStride);

    expect((resolution - 1) % outputStride).toEqual(0);
  });
});

describeWithFlags('util.toInputResolutionHeightAndWidth', ALL_ENVS, () => {
  function getExpectedResolution(
      inputShape: [number, number], outputStride: BodyPixOutputStride,
      expectedScalePercentage: number) {
    return inputShape.map(
        size => toValidInputResolution(
            size * expectedScalePercentage, outputStride));
  }

  it(`returns the full image size as a valid input resolution when ` +
         `internalResolution is 'full'`,
     () => {
       const inputShape: [number, number] = [1920, 1080];
       const outputStride = 16;
       const internalResolution = 'full';
       const expectedScalePercentage = 1.0;

       const expectedResult = getExpectedResolution(
           inputShape, outputStride, expectedScalePercentage);

       const result = toInputResolutionHeightAndWidth(
           internalResolution, outputStride, inputShape);

       expect(result).toEqual(expectedResult);
     });

  it(`returns 75% of the image size as a valid input resolution when ` +
         `internalResolution is 'high'`,
     () => {
       const inputShape: [number, number] = [400, 900];
       const outputStride = 16;
       const internalResolution = 'high';
       const expectedScalePercentage = 0.75;

       const expectedResult = getExpectedResolution(
           inputShape, outputStride, expectedScalePercentage);

       const result = toInputResolutionHeightAndWidth(
           internalResolution, outputStride, inputShape);

       expect(result).toEqual(expectedResult);
     });

  it(`returns 50% of the image size as a valid input resolution when ` +
         `internalResolution is 'medium'`,
     () => {
       const inputShape: [number, number] = [694, 309];
       const outputStride = 32;
       const internalResolution = 'medium';
       const expectedScalePercentage = 0.50;

       const expectedResult = getExpectedResolution(
           inputShape, outputStride, expectedScalePercentage);

       const result = toInputResolutionHeightAndWidth(
           internalResolution, outputStride, inputShape);

       expect(result).toEqual(expectedResult);
     });

  it(`returns 25% of the image size as a valid input resolution when ` +
         `internalResolution is 'low'`,
     () => {
       const inputShape: [number, number] = [930, 1001];
       const outputStride = 8;
       const internalResolution = 'low';
       const expectedScalePercentage = 0.25;

       const expectedResult = getExpectedResolution(
           inputShape, outputStride, expectedScalePercentage);

       const result = toInputResolutionHeightAndWidth(
           internalResolution, outputStride, inputShape);

       expect(result).toEqual(expectedResult);
     });

  it(`returns the {internalResolution}% of the image size as a valid input ` +
         `resolution when internalResolution is a number`,
     () => {
       const inputShape: [number, number] = [1450, 789];
       const outputStride = 16;
       const internalResolution = 0.675;

       const expectedResult =
           getExpectedResolution(inputShape, outputStride, internalResolution);

       const result = toInputResolutionHeightAndWidth(
           internalResolution, outputStride, inputShape);

       expect(result).toEqual(expectedResult);
     });

  it('does not raise an error when internalResolution is 2', () => {
    expect(() => {
      toInputResolutionHeightAndWidth(2.00, 16, [640, 480]);
    }).not.toThrow();
  });
  it('raises an error when internalResolution is larger than 2', () => {
    expect(() => {
      toInputResolutionHeightAndWidth(2.01, 16, [640, 480]);
    }).toThrow();
  });
  it('does not raise an error when internalResolution is 0.1', () => {
    expect(() => {
      toInputResolutionHeightAndWidth(0.1, 16, [640, 480]);
    }).not.toThrow();
  });
  it('raises an error when internalResolution is less than 0.1', () => {
    expect(() => {
      toInputResolutionHeightAndWidth(0.09, 16, [640, 480]);
    }).toThrow();
  });
});
