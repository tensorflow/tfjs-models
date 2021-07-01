import * as tf from '@tensorflow/tfjs-core';
import {textLineConfig} from '../interfaces';
import {Graph} from './graph';
import * as utils from '../utils';

export class TextProposalGraphBuilder{
	private textProposals: number[][] ;
	private boxesTable: number[][];
	private imSize: number[];
	private heights: number[];
	private scores: tf.Tensor;
	constructor() {
		this.textProposals= [[]];
		this.imSize = [];
		this.heights = [];
		this.scores = tf.tensor([]);
	}

	get_successions(index: number){

		const box = this.textProposals[index];
		const results=[];
		for(
			let left = Math.round(box[0])+1;
			left < Math.min(
				Math.round(box[0]) + textLineConfig.MAX_HORIZONTAL_GAP+1,
				this.imSize[1]
			);
			left++
		){
			const adjBoxIndices = this.boxesTable[left];
			for (const adjBoxIndex of adjBoxIndices){
				if(this.meet_v_iou(adjBoxIndex, index)) {
					results.push(adjBoxIndex);
				}
			}

			if(results.length!==0) {
				return results;
			}
		}
		return results;
	}

	get_precursors(index: number) {
		const box = this.textProposals[index];
		const results = [];

		for(let left = Math.round(box[0])-1;
			left > Math.max(
				Math.round(box[0] - textLineConfig.MAX_HORIZONTAL_GAP), 0) -1;
			left--){
			const adjBoxIndices = this.boxesTable[left];
			for (const adjBoxIndex of adjBoxIndices){
				if(this.meet_v_iou(adjBoxIndex, index)) {
					results.push(adjBoxIndex);
				}
			}
			if(results.length!==0) {
				return results;
			}
		}
		return results;
	}

	meet_v_iou(index1: number, index2: number){
		const overlapsV = (index1: number, index2: number) => {
			const h1 = this.heights[index1];
			const h2 = this.heights[index2];
			const y0 = Math.max(
				this.textProposals[index2][1], this.textProposals[index1][1]
			);
			const y1 = Math.min(
				this.textProposals[index2][3], this.textProposals[index1][3]
			);
			return Math.max(0, y1-y0+1)/Math.min(h1, h2);
		};

		const sizeSimilarity = (index1: number, index2: number) => {
			const h1 = this.heights[index1];
			const h2 = this.heights[index2];
			return Math.min(h1, h2) / Math.max(h1, h2);
		};

		return (
			overlapsV(index1, index2)>=textLineConfig.MIN_V_OVERLAPS &&
			sizeSimilarity(index1, index2)>=textLineConfig.MIN_SIZE_SIM
		);
	}

	is_succession_node(index: number, successionIndex: number) {
		const precursors = this.get_precursors(successionIndex);
		return tf.greaterEqual(
			tf.gather(this.scores,index), tf.max(tf.gather(this.scores,precursors)) )
			.arraySync();
	}

	build_graph<T extends tf.Tensor>(
		textProposals: T,
		scores: T,
		imSize: number[]
	){
		this.textProposals = textProposals.arraySync() as number[][];
		this.scores = utils.ravel(scores);
		this.imSize = imSize;
		const h1 = tf.reshape(
			tf.slice(textProposals,[0,3], [textProposals.shape[0],1]),
			[textProposals.shape[0]]
		);
		const h2 = tf.reshape(
			tf.slice(textProposals,[0,1], [textProposals.shape[0],1]),
			[textProposals.shape[0]]
		);
		this.heights = tf.add(tf.sub(h1,h2), 1).arraySync() as number[];
		const boxesTable =  Array.from(Array(imSize[1]), () => []);
		this.textProposals.forEach((item, index)=>{
			// @ts-ignore
			boxes_table[Math.round(item[0])].push(index);
		});
		this.boxesTable = boxesTable;
		const graph = tf.buffer(
			[textProposals.shape[0], textProposals.shape[0]], 'bool');
		for(let index = 0; index < this.textProposals.length; index++){
			const successions = this.get_successions(index);
			if(successions.length === 0) {
				continue;
			}

			let successionIndex;
			if (successions.length > 1) {
				successionIndex = successions[
					utils.argmax(tf.gather(this.scores,successions).arraySync() as number[])
					];
			}else {
				successionIndex = successions[0];
			}

			if (this.is_succession_node(index, successionIndex)) {
				graph.set(true, index, successionIndex);
			}

		}
		const _graph = graph.toTensor();
		return new Graph(_graph);
	}
}
