import * as tf from '@tensorflow/tfjs-core';

export interface PredictionConfig{
	nms_function: string;
	anchor_scales: number[];
	pixel_means: tf.Tensor;
	scales: number[];
	max_size:  number;
	has_rpn: boolean;
	detect_mode: string;
	pre_nms_topN: number;
	post_nms_topN: number;
	nms_thresh: number;
	min_size: number;
}
export const textLineConfig = {
	SCALE: 600,
	MAX_SCALE: 1200,
	TEXT_PROPOSALS_WIDTH: 16,
	MIN_NUM_PROPOSALS: 2,
	MIN_RATIO: 0.5,
	LINE_MIN_SCORE: 0.9,
	MAX_HORIZONTAL_GAP: 50,
	TEXT_PROPOSALS_MIN_SCORE: 0.7,
	TEXT_PROPOSALS_NMS_THRESH: 0.2,
	MIN_V_OVERLAPS: 0.7,
	MIN_SIZE_SIM: 0.7,
};
