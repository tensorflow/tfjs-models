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

// TODO(ardoerlemans): This filter should be merged with the existing
// one euro filter in pose_detection/src/calculators once PR #632 has been
// merged.

/**
 * Exponentially weighted moving average filter.
 * https://en.wikipedia.org/wiki/Moving_average
 */
class EWMA {
  private updateRate: number;
  private state: number[];

  constructor(updateRate: number) {
    this.updateRate = updateRate;
    this.state = [];
  }

  insert(observations: number[]) {
    if (!this.state.length) {
      this.state = observations.slice();
    } else {
      for (let obsIdx = 0; obsIdx < observations.length; obsIdx += 1) {
        this.state[obsIdx] = (1 - this.updateRate) * this.state[obsIdx] +
            this.updateRate * observations[obsIdx];
      }
    }
    return this.state;
  }
}

/**
 * One-euro filter that works on arrays of numbers.
 * https://hal.inria.fr/hal-00670496/document
 */
export class RobustOneEuroFilter {
  private updateRateOffset: number;
  private updateRateSlope: number;
  private fps: number;
  private thresholdOffset: number;
  private thresholdSlope: number;
  private prevObs: number[];
  private speed: number[];
  private speedEwma: EWMA;
  private threshold: number[];
  private state: number[];

  constructor(
      updateRateOffset = 40.0, updateRateSlope = 4e3, speedUpdateRate = 0.5,
      fps = 30, thresholdOffset = 0.5, thresholdSlope = 5.0) {
    this.updateRateOffset = updateRateOffset;
    this.updateRateSlope = updateRateSlope;
    this.fps = fps;
    this.thresholdOffset = thresholdOffset;
    this.thresholdSlope = thresholdSlope;

    this.prevObs = [];
    this.speed = [];
    this.speedEwma = new EWMA(speedUpdateRate);
    this.threshold = [];

    this.state = [];
  }

  insert(observations: number[], fps = 30) {
    if (fps > 0) {
      this.fps = fps;
    }
    this.threshold = observations.slice();
    for (let obsIdx = 0; obsIdx < observations.length; obsIdx += 1) {
      this.threshold[obsIdx] = this.thresholdOffset;
    }

    if (this.prevObs.length) {
      const rawDiff = observations.slice();
      for (let obsIdx = 0; obsIdx < observations.length; obsIdx += 1) {
        rawDiff[obsIdx] -= this.prevObs[obsIdx];
      }
      if (this.speed.length) {
        for (let obsIdx = 0; obsIdx < observations.length; obsIdx += 1) {
          this.threshold[obsIdx] = this.thresholdOffset +
              this.thresholdSlope * Math.abs(this.speed[obsIdx]);
        }
      }
      this.speed = this.speedEwma.insert(rawDiff);
    }
    this.prevObs = observations.slice();

    if (!this.state.length) {
      this.state = observations.slice();
    } else {
      for (let obsIdx = 0; obsIdx < observations.length; obsIdx += 1) {
        const updateRate = 1.0 /
            (1.0 +
             this.fps /
                 (this.updateRateOffset +
                  this.updateRateSlope * Math.abs(this.speed[obsIdx])));
        this.state[obsIdx] = this.state[obsIdx] +
            updateRate * this.threshold[obsIdx] *
                Math.asinh(
                    (observations[obsIdx] - this.state[obsIdx]) /
                    this.threshold[obsIdx]);
      }
    }
    return this.state;
  }
}
