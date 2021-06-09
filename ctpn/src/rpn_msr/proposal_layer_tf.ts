import * as tf from '@tensorflow/tfjs-core';
import * as utils from '../utils';
import {bbox_transform_inv, clip_boxes, filter_boxes} from '../rcnn/bbox_transform';
import {PredictionConfig} from '../interfaces';

export async function proposal_layer <T extends tf.Tensor4D, X extends tf.Tensor3D>(cfg: PredictionConfig, rpnClsProbReshape: T, rpnBboxPred:T, imInfo:X , test:string, _featStride = [16]){

	const _anchors = utils.generate_anchors( 16, [0.5, 1, 2], cfg.anchor_scales);
	const _numAnchors = _anchors.shape[0];
	if(rpnClsProbReshape.shape[0] !== 1) {
		throw new Error(`Only single item batches are supported`);
	}
	const [height, width] = [
		rpnClsProbReshape.shape[1], rpnClsProbReshape.shape[2]
	];

	const reshape = tf.reshape(
		rpnClsProbReshape, [1, height, width, _numAnchors, 2]
	);

	let scores = tf.reshape(
		tf.slice(reshape, [0,0,0,0,1],[1, height, width, _numAnchors,1]),
		[1, height, width, _numAnchors]);

	let bboxDeltas = rpnBboxPred.clone();
	let shiftX = tf.mul(tf.range(0, width), _featStride);
	let shiftY = tf.mul(tf.range(0, height), _featStride);
	[shiftX, shiftY] = tf.meshgrid(shiftX, shiftY);
	const shifts = tf.transpose(
		tf.stack(
			[
				utils.ravel(shiftX),
				utils.ravel(shiftY),
				utils.ravel(shiftX),
				utils.ravel(shiftY)
			], 0)
	);

	const A = _numAnchors;
	const K = shifts.shape[0];
	let anchors = tf.add(tf.reshape(_anchors,[1, A, 4]), tf.transpose(
		tf.reshape(shifts, [1, K, 4]), [1, 0, 2]
	) );

	anchors = tf.reshape(anchors,[K * A, 4]);

	// @ts-ignore
	bboxDeltas = tf.reshape(bboxDeltas,[-1, 4]);
	scores = tf.reshape(scores,[-1, 1]);
	anchors = tf.cast(anchors,'int32');
	// Convert anchors into proposals via bbox transformations
	let proposals = bbox_transform_inv(anchors, bboxDeltas);

	// proposals.print()
	// 2. clip predicted boxes to image

	// @ts-ignore
	proposals = clip_boxes(proposals, imInfo.arraySync()[0].slice(0,2));//
	let keep = await filter_boxes(proposals, cfg.min_size * 1);
	keep = utils.ravel(keep);
	proposals = tf.gather(proposals, tf.cast(keep,'int32'));
	scores = tf.gather(scores, tf.cast(keep,'int32'));
	bboxDeltas = tf.gather(bboxDeltas, tf.cast(keep,'int32'));
	
	let order = tf.reverse(utils.argSort(utils.ravel(scores)));
	if(cfg.pre_nms_topN > 0){
		order = tf.slice(order,0, cfg.pre_nms_topN);
	}
	proposals = tf.gather(proposals, tf.cast(order,'int32'));

	scores = tf.gather(scores, tf.cast(order,'int32'));

	bboxDeltas = tf.gather(bboxDeltas, tf.cast(order,'int32'));

	keep = await utils.nms(
		{
			dets: proposals as tf.Tensor2D,
			scores: scores as tf.Tensor2D,
			thresh:cfg.nms_thresh,
			method: cfg.nms_function}
	);

	if (cfg.post_nms_topN > 0 && keep.shape[0] > cfg.post_nms_topN){
		keep = tf.slice(keep,0, cfg.post_nms_topN);
	}
	proposals = tf.gather(proposals, tf.cast(keep,'int32'));
	scores = tf.gather(scores, tf.cast(keep,'int32'));
	bboxDeltas = tf.gather(bboxDeltas, tf.cast(keep,'int32'));
	return [
		utils.ravel(tf.cast(scores,'float32')),
		tf.cast(proposals,'float32'), bboxDeltas
	];
}
