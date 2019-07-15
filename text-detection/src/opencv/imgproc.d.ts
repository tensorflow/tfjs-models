import {Mat} from './Mat';
import {RotatedRect} from './RotatedRect';

interface Rect {}
export declare function minAreaRect(polytope: Mat): RotatedRect;
export declare function connectedComponents(
    graph: Mat, labels: Mat, connectivity: number): any;
