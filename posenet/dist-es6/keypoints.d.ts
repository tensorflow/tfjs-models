export declare type Tuple<T> = [T, T];
export declare type StringTuple = Tuple<string>;
export declare type NumberTuple = Tuple<number>;
export declare const partNames: string[];
export declare const NUM_KEYPOINTS: number;
export interface NumberDict {
    [jointName: string]: number;
}
export declare const partIds: NumberDict;
export declare const connectedPartIndeces: number[][];
