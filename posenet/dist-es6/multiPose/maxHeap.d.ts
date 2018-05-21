export declare class MaxHeap<T> {
    private priorityQueue;
    private numberOfElements;
    private getElementValue;
    constructor(maxSize: number, getElementValue: (element: T) => number);
    enqueue(x: T): void;
    dequeue(): T;
    empty(): boolean;
    size(): number;
    all(): T[];
    max(): T;
    private swim(k);
    private sink(k);
    private getValueAt(i);
    private less(i, j);
    private exchange(i, j);
}
