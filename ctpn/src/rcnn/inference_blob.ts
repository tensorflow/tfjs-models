import * as tf from '@tensorflow/tfjs-core';
import {argmax} from '../utils';
import {PredictionConfig} from '../interfaces';

interface BlobsInterface{
	data :  tf.Tensor4D | null;
	rois :  null | number | number[];
	im_info: tf.Tensor3D | null;
}

export function get_blobs(img: tf.Tensor3D, rois: number | null, cfg: PredictionConfig): [BlobsInterface, number[]]{
	const blobs: BlobsInterface = {
		data : null,
		rois : null,
		im_info: null
	};

	const [data, imScaleFactors] = _get_image_blob(img, cfg);
	blobs.data = data;
	return [blobs, imScaleFactors];
}

function _get_image_blob(im: tf.Tensor3D,
						 cfg: PredictionConfig): [tf.Tensor4D, number[]]{

	let imOrig = tf.cast(im,'float32');
	imOrig = tf.sub(imOrig, cfg.pixel_means);

	const imShape = imOrig.shape;
	const [w, h] = imShape.slice(0,2);
	const imSizeMin = Math.min(w, h);
	const imSizeMax = Math.max(w,h);
	const processedIms = [];
	const imScaleFactors = [];
	for (const targetSize of cfg.scales){
		let imScale = targetSize / imSizeMin;
		// Prevent the biggest axis from being more than max_size
		if (Math.round(imScale * imSizeMax) > cfg.max_size){
			imScale = cfg.max_size / imSizeMax;
		}
		im = tf.image.resizeBilinear(imOrig, [~~(w * imScale), ~~(h * imScale)]);
		imScaleFactors.push(imScale);
		processedIms.push(im);
	}
	// Create a blob to hold the input images
	const blob = im_list_to_blob(processedIms);
	return [blob, imScaleFactors];
}

function im_list_to_blob(ims: tf.Tensor3D[]): tf.Tensor4D{

	// Convert a list of images into a network input.
	//
	// Assumes images are already prepared (means subtracted, BGR order, ...).
	const index = argmax(ims.map((im) => im.shape[0] * im.shape[1] * im.shape[2]));
	const maxShape = ims[index].shape;

	const numImages = ims.length;
	const blob = tf.zeros([numImages, maxShape[0], maxShape[1], 3], 'float32')
		.arraySync() as number[][][][];

	for (let i = 0;i < numImages; i++){
		const im = ims[i];
		blob[i] = im.arraySync();
	}
	return tf.tensor(blob);
}
