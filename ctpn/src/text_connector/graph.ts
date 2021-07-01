import * as tf from '@tensorflow/tfjs-core';

type graphType = tf.Tensor | tf.Tensor1D | tf.Tensor2D;

export class Graph{
	graph: graphType;
	constructor(graph: graphType) {
		this.graph = graph;
	}

	async sub_graphs_connected(){
		const subGraphs = [];
		for (let index = 0; index < this.graph.shape[0]; index++){
			const firstCondition = tf.logicalNot(
				tf.any(
					tf.reshape(
						tf.slice(this.graph,[0,index], [this.graph.shape[0],1]),
						[this.graph.shape[0]]
					)
				)
			);
			const secondCondition = tf.any(
				tf.reshape(
					tf.slice(this.graph,[index,0], [1 ,this.graph.shape[0]]),
					[this.graph.shape[0]]
				)
			);
			const condition = tf.logicalAnd(firstCondition, secondCondition).arraySync();

			if(condition) {
				let v: number | tf.Tensor3D | tf.Tensor1D | tf.Tensor2D = index;
				subGraphs.push([v]);

				while (tf.any(tf.gather(this.graph, v)).arraySync()){
					v = await tf.whereAsync(tf.gather(this.graph, v));
					v = v.arraySync()[0][0];
					subGraphs[subGraphs.length-1].push(v);
				}
			}
		}
		return subGraphs;
	}
}
