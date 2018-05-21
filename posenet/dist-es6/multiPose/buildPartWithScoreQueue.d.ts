import { PartWithScore, TensorBuffer3D } from '../types';
import { MaxHeap } from './maxHeap';
export declare function buildPartWithScoreQueue(scoreThreshold: number, localMaximumRadius: number, scores: TensorBuffer3D): MaxHeap<PartWithScore>;
