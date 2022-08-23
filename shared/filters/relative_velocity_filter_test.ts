/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {RelativeVelocityFilter} from './relative_velocity_filter';

describe('Relative velocity filter ', () => {
  it('Smoke.', () => {
    const filter =
        new RelativeVelocityFilter({windowSize: 1, velocityScale: 1});

    const timestamp1 = 1;

    expect(filter.apply(95.5, timestamp1, 0.5)).toBe(95.5);
    expect(filter.apply(200.5, timestamp1, 0.5)).toBe(200.5);
    expect(filter.apply(1000.5, timestamp1, 0.5)).toBe(1000.5);
    expect(filter.apply(2000, timestamp1, 0.5)).toBe(2000);
  });

  it('Same value scale different velocity scales legacy.', () => {
    // More sensitive filter.
    const filter1 =
        new RelativeVelocityFilter({windowSize: 5, velocityScale: 45});

    // Less sensitive filter.
    const filter2 =
        new RelativeVelocityFilter({windowSize: 5, velocityScale: 0.1});

    let result1;
    let result2;
    let value;
    const valueScale = 1;

    value = 1;
    result1 = filter1.apply(value, 1000 /* 1ms */, valueScale);
    result2 = filter2.apply(value, 1000 /* 1ms */, valueScale);
    expect(result1).toEqual(result2);

    value = 10;
    result1 = filter1.apply(value, 2000 /* 2ms */, valueScale);
    result2 = filter2.apply(value, 2000 /* 2ms */, valueScale);
    expect(result1).toBeGreaterThan(result2);

    value = 2;
    result1 = filter1.apply(value, 3000 /* 3ms */, valueScale);
    result2 = filter2.apply(value, 3000 /* 3ms */, valueScale);
    expect(result1).toBeLessThan(result2);

    value = 20;
    result1 = filter1.apply(value, 4000 /* 4ms */, valueScale);
    result2 = filter2.apply(value, 4000 /* 4ms */, valueScale);
    expect(result1).toBeGreaterThan(result2);

    value = 10;
    result1 = filter1.apply(value, 5000 /* 5ms */, valueScale);
    result2 = filter2.apply(value, 5000 /* 5ms */, valueScale);
    expect(result1).toBeLessThan(result2);

    value = 50;
    result1 = filter1.apply(value, 6000 /* 6ms */, valueScale);
    result2 = filter2.apply(value, 6000 /* 6ms */, valueScale);
    expect(result1).toBeGreaterThan(result2);

    value = 30;
    result1 = filter1.apply(value, 7000 /* 7ms */, valueScale);
    result2 = filter2.apply(value, 7000 /* 7ms */, valueScale);
    expect(result1).toBeLessThan(result2);
  });

  it('Different constant value scales same velocity scale legacy.', () => {
    const sameVelocityScale = 1;
    const filter1 = new RelativeVelocityFilter(
        {windowSize: 3, velocityScale: sameVelocityScale});
    const filter2 = new RelativeVelocityFilter(
        {windowSize: 3, velocityScale: sameVelocityScale});

    let result1;
    let result2;
    let value;
    // smaller value scale will decrease cumulative speed and alpha so with
    // smaller scale and same other params filter will believe new values
    // a little bit less.
    const valueScale1 = 0.5;
    const valueScale2 = 1;

    value = 1;
    result1 = filter1.apply(value, 1000 /* 1ms */, valueScale1);
    result2 = filter2.apply(value, 1000 /* 1ms */, valueScale2);
    expect(result1).toEqual(result2);

    value = 10;
    result1 = filter1.apply(value, 2000 /* 2ms */, valueScale1);
    result2 = filter2.apply(value, 2000 /* 2ms */, valueScale2);
    expect(result1).toBeLessThan(result2);

    value = 2;
    result1 = filter1.apply(value, 3000 /* 3ms */, valueScale1);
    result2 = filter2.apply(value, 3000 /* 3ms */, valueScale2);
    expect(result1).toBeGreaterThan(result2);

    value = 20;
    result1 = filter1.apply(value, 4000 /* 4ms */, valueScale1);
    result2 = filter2.apply(value, 4000 /* 4ms */, valueScale2);
    expect(result1).toBeLessThan(result2);
  });

  it('Translation invariance.', () => {
    const originalDataPoints = [
      {value: 1, scale: 0.5}, {value: 10, scale: 5}, {value: 20, scale: 10},
      {value: 30, scale: 15}, {value: 40, scale: 0.5}, {value: 50, scale: 0.5},
      {value: 60, scale: 5}, {value: 70, scale: 10}, {value: 80, scale: 15},
      {value: 90, scale: 5}, {value: 70, scale: 10}, {value: 50, scale: 15},
      {value: 80, scale: 15}
    ];

    // The amount by which the input values are uniformly translated.
    const valueOffset = 100;

    // The uniform time delta.
    const timeDelta = 1000; /* 1ms */

    // The filter parameters are the same between the two filters.
    const windowSize = 5;
    const velocityScale = 0.1;

    // Perform the translation.
    const translatedDataPoints = [];
    for (const dp of originalDataPoints) {
      translatedDataPoints.push(
          {value: dp.value + valueOffset, scale: dp.scale});
    }

    const originalPointsFilter =
        new RelativeVelocityFilter({windowSize, velocityScale});
    const translatedPointsFilter =
        new RelativeVelocityFilter({windowSize, velocityScale});

    // The minimal difference which is considered a divergence.
    const divergenceGap = 0.001;

    // The amount of the times this gap is achieved with legacy transition.
    // Note that on the first iteration the filters should output the unfiltered
    // input values, so no divergence should occur.
    // This amount obviously depends on the values in `originalDataPoints`, so
    // should be changed accordingly when they are updated.
    const divergenceTimes = 5;

    // The minimal difference which is considered a large divergence.
    const largeDivergenceGap = 10;

    // The amount of times it is achieved.
    // This amount obviously depends on the values in `originalDataPoints`, so
    // should be changed accordingly when they are updated.
    const largeDivergenceTimes = 1;

    let timesDiverged = 0;
    let timesLargelyDiverged = 0;
    let timestamp = 0;
    for (let iteration = 0; iteration < originalDataPoints.length;
         ++iteration, timestamp += timeDelta) {
      const originalDataPoint = originalDataPoints[iteration];
      const filteredOriginalValue = originalPointsFilter.apply(
          originalDataPoint.value, timestamp, originalDataPoint.scale);
      const translatedDataPoint = translatedDataPoints[iteration];
      const actualFilteredTranslatedValue = translatedPointsFilter.apply(
          translatedDataPoint.value, timestamp, translatedDataPoint.scale);

      const expectedFilteredTranslatedValue =
          filteredOriginalValue + valueOffset;

      const difference = Math.abs(
          actualFilteredTranslatedValue - expectedFilteredTranslatedValue);

      if (iteration === 0) {
        // On the first iteration, the unfiltered values are returned.
        expect(filteredOriginalValue).toEqual(originalDataPoint.value);
        expect(actualFilteredTranslatedValue)
            .toEqual(translatedDataPoint.value);
        expect(difference).toEqual(0);
      } else {
        if (difference >= divergenceGap) {
          ++timesDiverged;
        }
        if (difference >= largeDivergenceGap) {
          ++timesLargelyDiverged;
        }
      }
    }

    expect(timesDiverged).toBeGreaterThanOrEqual(divergenceTimes);
    expect(timesLargelyDiverged).toBeGreaterThanOrEqual(largeDivergenceTimes);
  });
});
