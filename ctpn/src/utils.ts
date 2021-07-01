import * as tf from '@tensorflow/tfjs-core';

export function bgr2rgb<T extends tf.Tensor>(image: T){
	return tf.reverse(image, -1);
}

export function resize_im(im: tf.Tensor3D, scale: number, maxScale: number |null): [tf.Tensor3D, number] {
	let f = scale / Math.min(im.shape[0], im.shape[1]);
	if (maxScale != null && f * Math.max(im.shape[0], im.shape[1]) > maxScale) {
		f = maxScale / Math.max(im.shape[0], im.shape[1]);
	}
	const [newH, newW] = [ ~~(im.shape[0] * f), ~~(im.shape[1] * f)];
	return [tf.image.resizeBilinear(im, [newH, newW]), f];
}

export function argmax(array: number[]): number{
	return [].reduce.call(array, (_m: unknown,
								  _c: never,
								  _i: number,
								  _arr: never[]) =>
		_c > _arr[_m as number] ? _i : _m, 0) as number;
}

export function ravel<T extends tf.Tensor>(tensor: T){
	const array = tensor.arraySync() as number[]; //mistaken?
	return tf.tensor(tf.util.flatten(array));
}

export function argSort <T extends tf.Tensor>(tensor: T){
	const array = tensor.arraySync() as number[];
	const initial = Array.from(array);
	const sorted = array.sort((a, b)=> a-b);
	const args = sorted.map( item=>initial.indexOf(item));
	return tf.tensor1d(args);
}

interface NMSCfg{
	dets: tf.Tensor2D;
	scores: tf.Tensor2D;
	thresh: number;
	method: string;
}

export async function nms(config: NMSCfg){
	return (config.method==='TF')?
		tf_nms(config.dets, config.scores, config.thresh)
		:
		authNMS(config.dets, config.scores, config.thresh);
}

function tf_nms<T extends tf.Tensor2D>(dets: T, scores: T, thresh: number){
	return tf.image.nonMaxSuppression(dets, ravel(scores) as tf.Tensor1D,
		2000, 0.2,thresh);
}

async function authNMS<T extends tf.Tensor2D>(dets: T,
											  scores: T,
											  thresh: number
){

	const x1 = tf.squeeze(tf.slice(dets,[0,0], [dets.shape[0],1]));
	const y1 = tf.squeeze(tf.slice(dets,[0,1], [dets.shape[0],1]));
	const x2 = tf.squeeze(tf.slice(dets,[0,2], [dets.shape[0],1]));
	const y2 = tf.squeeze(tf.slice(dets,[0,3], [dets.shape[0],1]));

	const areas = tf.mul( tf.add(tf.sub(x2,x1), 1), tf.add(tf.sub(y2,y1), 1)) ;

	let order = tf.reverse(argSort(scores));//order = scores.argsort()[::-1]

	order = ravel(order) as tf.Tensor1D;
	let keep = tf.tensor1d([]);
	while(order.shape[0] > 0){
		const i = tf.cast(tf.slice(order,0,1),'int32');
		keep = tf.concat([keep, tf.cast(i,'float32')]);
		const xx1 = tf.maximum(
			tf.gather(x1,i), tf.gather(x1, tf.cast(tf.slice(order,1,-1), 'int32') )
		);
		const yy1 = tf.maximum(
			tf.gather(y1, i), tf.gather(y1, tf.cast(tf.slice(order,1,-1), 'int32') )
		);
		const xx2 = tf.minimum(
			tf.gather(x2, i), tf.gather(x2, tf.cast(tf.slice(order,1,-1), 'int32') )
		);
		const yy2 = tf.minimum(
			tf.gather(y2,i), tf.gather(y2, tf.cast(tf.slice(order,1,-1),'int32') )
		);
		const w = tf.maximum(0.0, tf.add(tf.sub(xx2,xx1),1) );
		const h = tf.maximum(0.0, tf.add(tf.sub(yy2, yy1),1) );
		const inter = tf.mul(w,h);
		const ovr = tf.div(inter, ( tf.sub((tf.add(tf.gather(areas,i),
			tf.gather(areas,tf.cast(tf.slice(order,1,-1),'int32')))),inter) ));
		let inds = await tf.whereAsync( tf.lessEqual(ovr,thresh) );
		inds = ravel(inds) as tf.Tensor2D;
		order = tf.gather(order, tf.cast(tf.add(inds,1),'int32') );

	}

	return keep;
}

function scale_anchor(anchor: number[], h: number, w: number): number[] {
	const xCtr = (anchor[0] + anchor[2]) * 0.5;
	const yCtr = (anchor[1] + anchor[3]) * 0.5;

	const scaledAnchor = Array.from(anchor);
	scaledAnchor[0] = ~~(xCtr - w / 2);
	// xmin
	scaledAnchor[2] = ~~(xCtr + w / 2);
	// xmax
	scaledAnchor[1] = ~~(yCtr - h / 2);
	// ymin
	scaledAnchor[3] = ~~(yCtr + h / 2);
	// ymax
	return scaledAnchor;
}

function generate_basic_anchors(sizes: number[][], baseSize= 16) {
	const baseAnchor = [0, 0, baseSize - 1, baseSize - 1];
	const anchors = tf.buffer([sizes.length, 4], 'int32');
	let index = 0;
	for (const item of sizes) {
		const [x, y, z ,w] = scale_anchor(baseAnchor, item[0], item[1]);
		anchors.set(x, index, 0);
		anchors.set(y, index, 1);
		anchors.set(z, index, 2);
		anchors.set(w, index, 3);
		index += 1;
	}

	return anchors.toTensor();
}
export function generate_anchors(baseSize= 16, ratios= [0.5, 1, 2], scales= [3**2,4**2,5**2,6**2]) {
	const heights = tf.tensor1d([11, 16, 23, 33, 48, 68, 97, 139, 198, 283]);
	const widths = tf.tensor1d([baseSize]);
	const sizes = tf.stack([heights, tf.tile(widths,heights.shape)], 1);

	return generate_basic_anchors(sizes.arraySync() as number[][]);
}
