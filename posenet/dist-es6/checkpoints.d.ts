import { ConvolutionDefinition } from './mobilenet';
export declare type Checkpoint = {
    url: string;
    architecture: ConvolutionDefinition[];
};
export declare const checkpoints: {
    [multiplier: number]: Checkpoint;
};
