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

class EWMA {
  private update_rate: number;
  private state: number[];

	constructor(update_rate: number) {
		this.update_rate = update_rate;
		this.state = [];
	}

	insert(observations: number[]) {
		if (!this.state.length) {
			this.state = observations.slice();
		} else {
			for (var obs_idx = 0; obs_idx < observations.length; obs_idx += 1) {
				this.state[obs_idx] =
					(1 - this.update_rate) * this.state[obs_idx] +
					this.update_rate * observations[obs_idx];
			}
		}
		return this.state;
	}
}

export class OneEuroFilter {
  private update_rate_offset: number;
  private update_rate_slope: number;
  private fps: number;
  private threshold_offset: number;
  private threshold_slope: number;
  private prev_obs: number[];
  private speed: number[];
  private speed_ewma: EWMA;
  private threshold: number[];
  private state: number[];

	constructor(update_rate_offset=40.0, update_rate_slope=4e3, speed_update_rate=0.5, fps=30,
	            threshold_offset=0.5, threshold_slope=5.0) {
		this.update_rate_offset = update_rate_offset;
		this.update_rate_slope = update_rate_slope;
		this.fps = fps;
		this.threshold_offset = threshold_offset;
		this.threshold_slope = threshold_slope;

		this.prev_obs = [];
		this.speed = [];
		this.speed_ewma = new EWMA(speed_update_rate);
		this.threshold = [];

		this.state = [];
	}

	insert(observations: number[], fps=30) {
		if (fps > 0) {
			this.fps = fps;
		}
    this.threshold = observations.slice();
    for (var obs_idx = 0; obs_idx < observations.length; obs_idx += 1) {
    	this.threshold[obs_idx] = this.threshold_offset;
    }

		if (this.prev_obs.length) {
			var raw_diff = observations.slice();
			for (var obs_idx = 0; obs_idx < observations.length; obs_idx += 1) {
				raw_diff[obs_idx] -= this.prev_obs[obs_idx];
			}
      if (this.speed.length) {
      	for (var obs_idx = 0; obs_idx < observations.length; obs_idx += 1) {
        	this.threshold[obs_idx] = this.threshold_offset + this.threshold_slope * Math.abs(this.speed[obs_idx]);
        }
      }
			this.speed = this.speed_ewma.insert(raw_diff);
		}
		this.prev_obs = observations.slice();

		if (!this.state.length) {
			this.state = observations.slice();
		} else {
			for (var obs_idx = 0; obs_idx < observations.length; obs_idx += 1) {
				var update_rate = 1.0 /
					(1.0 + this.fps / (
						this.update_rate_offset + this.update_rate_slope * Math.abs(this.speed[obs_idx])));
        this.state[obs_idx] = this.state[obs_idx] + update_rate * this.threshold[obs_idx] * Math.asinh(
        	(observations[obs_idx] - this.state[obs_idx]) / this.threshold[obs_idx]);
			}
		}
		return this.state;
	}
}
