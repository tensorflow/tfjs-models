function half(k) {
    return Math.floor(k / 2);
}
var MaxHeap = (function () {
    function MaxHeap(maxSize, getElementValue) {
        this.priorityQueue = new Array(maxSize);
        this.numberOfElements = -1;
        this.getElementValue = getElementValue;
    }
    MaxHeap.prototype.enqueue = function (x) {
        this.priorityQueue[++this.numberOfElements] = x;
        this.swim(this.numberOfElements);
    };
    MaxHeap.prototype.dequeue = function () {
        var max = this.priorityQueue[0];
        this.exchange(0, this.numberOfElements--);
        this.sink(0);
        this.priorityQueue[this.numberOfElements + 1] = null;
        return max;
    };
    MaxHeap.prototype.empty = function () {
        return this.numberOfElements === -1;
    };
    MaxHeap.prototype.size = function () {
        return this.numberOfElements + 1;
    };
    MaxHeap.prototype.all = function () {
        return this.priorityQueue.slice(0, this.numberOfElements + 1);
    };
    MaxHeap.prototype.max = function () {
        return this.priorityQueue[0];
    };
    MaxHeap.prototype.swim = function (k) {
        while (k > 0 && this.less(half(k), k)) {
            this.exchange(k, half(k));
            k = half(k);
        }
    };
    MaxHeap.prototype.sink = function (k) {
        while (2 * k <= this.numberOfElements) {
            var j = 2 * k;
            if (j < this.numberOfElements && this.less(j, j + 1)) {
                j++;
            }
            if (!this.less(k, j)) {
                break;
            }
            this.exchange(k, j);
            k = j;
        }
    };
    MaxHeap.prototype.getValueAt = function (i) {
        return this.getElementValue(this.priorityQueue[i]);
    };
    MaxHeap.prototype.less = function (i, j) {
        return this.getValueAt(i) < this.getValueAt(j);
    };
    MaxHeap.prototype.exchange = function (i, j) {
        var t = this.priorityQueue[i];
        this.priorityQueue[i] = this.priorityQueue[j];
        this.priorityQueue[j] = t;
    };
    return MaxHeap;
}());
export { MaxHeap };
//# sourceMappingURL=maxHeap.js.map