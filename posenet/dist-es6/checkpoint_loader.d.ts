import { Tensor } from '@tensorflow/tfjs';
export interface CheckpointVariable {
    filename: string;
    shape: number[];
}
export declare type CheckpointManifest = {
    [varName: string]: CheckpointVariable;
};
export declare class CheckpointLoader {
    private urlPath;
    private checkpointManifest;
    private variables;
    constructor(urlPath: string);
    private loadManifest();
    getCheckpointManifest(): Promise<CheckpointManifest>;
    getAllVariables(): Promise<{
        [varName: string]: Tensor;
    }>;
    getVariable(varName: string): Promise<Tensor>;
}
