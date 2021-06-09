import * as tf from '@tensorflow/tfjs-core';

export function bbox_transform_inv<T extends tf.Tensor>(boxes: T, deltas: T){

	boxes = tf.cast(boxes,deltas.dtype);

	const w1 = tf.reshape(
		tf.slice(boxes,[0,2], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const w2 = tf.reshape(
		tf.slice(boxes,[0,0], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const h1 = tf.reshape(
		tf.slice(boxes,[0,3], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const h2 = tf.reshape(
		tf.slice(boxes,[0,1], [boxes.shape[0],1]),[boxes.shape[0]]
	);

	const widths = tf.add(tf.sub(w1, w2), 1);
	const heights = tf.add(tf.sub(h1,h2), 1);

	const ctrX = tf.add(w2,tf.mul(0.5,widths));
	const ctrY = tf.add(h2,tf.mul(0.5,heights));

	//const dx = deltas.slice([0,0], [deltas.shape[0],1]);
	const dy = tf.slice(deltas,[0,1], [deltas.shape[0],1]);
	//const dw = deltas.slice([0,2], [deltas.shape[0],1]);
	const dh = tf.slice(deltas,[0,3], [deltas.shape[0],1]);

	const predCtrX = tf.reshape(ctrX,[ctrX.shape[0],1]);
	const predCtrY = tf.add(
		tf.mul( dy,tf.reshape(heights,[heights.shape[0],1]) ),
		tf.reshape(ctrY,[ctrY.shape[0],1])
	);
	const predW = tf.reshape(widths,[widths.shape[0],1]);
	const predH = tf.mul( tf.exp(dh), tf.reshape(heights,[heights.shape[0],1]));

	const x1 = tf.sub(predCtrX,tf.mul(0.5, predW) );
	const y1 = tf.sub(predCtrY,tf.mul(0.5, predH) );
	const x2 = tf.add(predCtrX,tf.mul(0.5, predW) );
	const y2 = tf.add(predCtrY,tf.mul(0.5, predH) );

	return tf.reshape(tf.stack( [x1, y1, x2, y2], -2 ),deltas.shape);
}

export function clip_boxes<T extends tf.Tensor, X extends number[]>(boxes: T, imShape: X){
// Clip boxes to image boundaries.
// // x1 >= 0
	const b1 = tf.maximum(
		tf.minimum (tf.slice(boxes,[0,0], [boxes.shape[0],1]), imShape[1] -1), 0);
// // y1 >= 0
	const b2 = tf.maximum(
		tf.minimum (tf.slice(boxes,[0,1], [boxes.shape[0],1]), imShape[0] -1), 0);
// // x2 < im_shape[1]
	const b3 = tf.maximum(
		tf.minimum (tf.slice(boxes,[0,2], [boxes.shape[0],1]), imShape[1] -1), 0);
// // y2 < im_shape[0]
	const b4 = tf.maximum(
		tf.minimum (tf.slice(boxes,[0,3], [boxes.shape[0],1]), imShape[0] -1), 0);

	return tf.reshape(tf.stack( [b1, b2, b3, b4], -2 ),boxes.shape);

}

export function filter_boxes<T extends tf.Tensor, X extends number>(boxes: T, minSize: X): Promise<tf.Tensor>{
// Remove all boxes with any side smaller than min_size.
	const _wsPartOne = tf.reshape(
		tf.slice(boxes, [0,2],[boxes.shape[0],1]),[boxes.shape[0]]
	);
	const _wsPartTwo =  tf.reshape(
		tf.slice(boxes, [0,0], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const ws = tf.add(tf.sub(_wsPartOne,_wsPartTwo),1);

	const _hsPartOne = tf.reshape(
		tf.slice(boxes, [0,3], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const _hsPartTwo = tf.reshape(
		tf.slice(boxes, [0,1], [boxes.shape[0],1]),[boxes.shape[0]]
	);
	const hs = tf.add(tf.sub(_hsPartOne,_hsPartTwo),1);

	const condPartOne = tf.greaterEqual(ws, minSize);
	const condPartTwo = tf.greaterEqual(hs, minSize);

	const bitwise = tf.logicalAnd(condPartOne, condPartTwo);

	return tf.whereAsync( bitwise ); // different result
}
