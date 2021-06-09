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

import * as tf from '@tensorflow/tfjs-core';
import * as tfconv from '@tensorflow/tfjs-converter';
import * as utils from './utils';
import {get_blobs} from './rcnn/inference_blob';
import {proposal_layer} from './rpn_msr/proposal_layer_tf';
import {TextDetector} from './text_connector/detectors';
import {PredictionConfig} from './interfaces';
const IMAGE_SIZE = 600;
const MAX_SCALE = 1200;

export interface CTPN {
	load(): Promise<void>;
	predict(img: HTMLImageElement, config: PredictionConfig):
		Promise<{prediction: tf.Tensor, scalefactor: number}>;
	draw <T extends tf.Tensor>(
		canvas: HTMLCanvasElement,
		_boxes: T,
		scale: number,
		color: string): void;
}

export async function load(modelUrl: string): Promise<CTPN> {
	if (tf == null) {
		throw new Error(
			`Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
			`also include @tensorflow/tfjs on the page before using this model.`);
	}
	if (!modelUrl) {
		throw new Error(
			`Invalid url for CTPN model. Valid versions are: ` +
			`https://cdn.jsdelivr.net/gh/BadMachine/tfjs-text-detection-ctpn/ctpn_web/model.json`);
	}

	// User provides versionStr / alphaStr.

	// User provides modelUrl & optional<inputRange>.

	const ctpn = new CTPNImpl(modelUrl);
	await ctpn.load();
	return ctpn;
}

class CTPNImpl implements CTPN {

	model: tfconv.GraphModel;
	constructor(public modelUrl: string) {

	}
	async load() {
		this.model = await tfconv.loadGraphModel(this.modelUrl);
		// Warmup the model.
		const result = tf.tidy(
			() => this.model.predict(tf.zeros(
				[1, IMAGE_SIZE, IMAGE_SIZE, 3]))) as tf.Tensor;
		await result.data();
		result.dispose();
	}
	async predict(img: HTMLImageElement, config: PredictionConfig):
		Promise<{prediction: tf.Tensor, scalefactor: number}>{

		const origin = tf.browser.fromPixels(img);
		const originBGR = utils.bgr2rgb(origin);
		const [image, scale] = utils.resize_im(originBGR, IMAGE_SIZE, MAX_SCALE);

		const [blobs, imScales] = get_blobs(image, null, config);
		const imBlob = blobs.data;
		if (config.has_rpn){
			blobs.im_info =
				tf.tensor( [[imBlob.shape[1], imBlob.shape[2], imScales[0]]]);
		}
		const model = await this.model;
		const raw = await model.executeAsync(tf.expandDims(image));
		const [clsProb, boxPred] = raw as tf.Tensor4D[];
		const [scores, proposals] = await proposal_layer(
			config,
			clsProb,
			boxPred,
			blobs.im_info,
			'TEST'
		);
		const boxes = tf.div(proposals, imScales[0]);
		const textDetector = new TextDetector(config);
		const _boxes = await textDetector.detect(
			boxes, tf.reshape(scores,[scores.shape[0],1]), image.shape.slice(0,2)
		);
		return {prediction:_boxes, scalefactor:scale};
	}

	draw <T extends tf.Tensor>(
		canvas: HTMLCanvasElement,
		_boxes: T,
		scale: number,
		color: string): void{

	}
}
