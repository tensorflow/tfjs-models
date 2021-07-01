import * as tf from '@tensorflow/tfjs-core';
import {TextProposalConnectorOriented} from './text_proposal_connector_oriented';
import {PredictionConfig, textLineConfig} from '../interfaces';
import * as utils from '../utils';

export class TextDetector{
	mode: string;
	NMS_FUNCTION: string;
	textProposalConnector: TextProposalConnectorOriented;
	constructor(cfg: PredictionConfig){
		this.mode = cfg.detect_mode;
		this.NMS_FUNCTION = cfg.detect_mode;
		this.textProposalConnector = new TextProposalConnectorOriented();
	}

	async detect<T extends tf.Tensor>(
		textProposals: T, scores: T, size: number[]
	){

		const scoresFlat = utils.ravel(scores);
		let keepInds: tf.Tensor = await tf.whereAsync(
			tf.greater(scoresFlat,textLineConfig.TEXT_PROPOSALS_MIN_SCORE)
		);
		keepInds = tf.cast(utils.ravel(keepInds),'int32');
		[textProposals, scores] = [
			tf.gather(textProposals, keepInds), tf.gather(scores, keepInds)
		];
		const sortedIndices = tf.cast(
			tf.reverse(utils.argSort(utils.ravel(scores))),'int32');
		[textProposals, scores] = [
			tf.gather(textProposals, sortedIndices), tf.gather(scores, sortedIndices)
		];
		keepInds = await utils.nms({
			dets: textProposals as tf.Tensor2D,
			scores: scores as tf.Tensor2D,
			thresh: textLineConfig.TEXT_PROPOSALS_NMS_THRESH,
			method: this.NMS_FUNCTION
		});

		keepInds = tf.cast(keepInds,'int32');
		[textProposals, scores] = [
			tf.gather(textProposals, keepInds), tf.gather(scores, keepInds)
		];

		const textRecs = await this.textProposalConnector.get_text_lines(
			textProposals, scores, size
		);
		keepInds = await this.filter_boxes(textRecs);
		keepInds = tf.unstack(keepInds,1)[0];
		return tf.gather(textRecs,keepInds);

	}

	async filter_boxes<T extends tf.Tensor>(boxes: T){

		const heights = tf.buffer( [boxes.shape[0], 1] );

		const widths = tf.buffer( [boxes.shape[0], 1] );

		const scores = tf.buffer( [boxes.shape[0], 1] );

		let index = 0;
		for(let i = 0; i < boxes.shape[0]; i++){
			heights.set(
				tf.add(
					tf.div(
						tf.add(
							tf.abs(
								tf.sub(tf.gatherND(boxes,[i, 5]),
									tf.gatherND(boxes,[i, 1]))
							),
							tf.abs(
								tf.sub(
									tf.gatherND(boxes,[i, 7]),
									tf.gatherND(boxes, [i, 3]) ))
						),2.0),1).arraySync() as number
				, 0,index);

			widths.set( tf.add(
				tf.div(
					tf.add(
						tf.abs(
							tf.sub(tf.gatherND(boxes,[i, 2]),
								tf.gatherND(boxes,[i, 0]))),
						tf.abs(
							tf.sub(
								tf.gatherND(boxes,[i, 6]),tf.gatherND(boxes, [i, 4]) ))
					),2.0),1).arraySync() as number, 0,index);

			scores.set(tf.gatherND(boxes,[i, 8]).arraySync() as number, 0,index);

			index+=1;
		}
		const _heights = heights.toTensor();
		const _widths = widths.toTensor();
		const _scores = scores.toTensor();
		return tf.whereAsync(
			tf.logicalAnd(tf.logicalAnd( tf.greater(
				tf.div(_widths,_heights), textLineConfig.MIN_RATIO),
				tf.greater(_scores,textLineConfig.LINE_MIN_SCORE) ),
				tf.greater(_widths,tf.mul(
					textLineConfig.TEXT_PROPOSALS_WIDTH,
					textLineConfig.MIN_NUM_PROPOSALS)) ) );
	}
}
