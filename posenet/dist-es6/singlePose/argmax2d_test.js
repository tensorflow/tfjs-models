import * as tf from '@tensorflow/tfjs';
import { argmax2d } from './argmax2d';
describe('argmax2d', function () {
    it('x = [2, 2, 1]', function () {
        var input = tf.tensor3d([1, 2, 0, 3], [2, 2, 1]);
        var result = argmax2d(input);
        var expectedResult = tf.tensor2d([1, 1], [1, 2], 'int32');
        tf.test_util.expectArraysClose(result, expectedResult);
    });
    it('x = [3, 3, 1]', function () {
        var input1 = tf.tensor3d([1, 2, 0, 3, 4, -1, 2, 9, 6], [3, 3, 1]);
        var input2 = tf.tensor3d([.5, .2, .9, 4.3, .2, .7, .6, -0.11, 1.4], [3, 3, 1]);
        tf.test_util.expectArraysClose(argmax2d(input1), tf.tensor2d([2, 1], [1, 2], 'int32'));
        tf.test_util.expectArraysClose(argmax2d(input2), tf.tensor2d([1, 0], [1, 2], 'int32'));
    });
    it('x = [3, 3, 3]', function () {
        var input1 = tf.tensor3d([1, 2, 0, 3, 4, -1, 2, 9, 6], [3, 3, 1]);
        var input2 = tf.tensor3d([.5, .2, .9, 4.3, .2, .7, .6, -.11, 1.4], [3, 3, 1]);
        var input3 = tf.tensor3d([4, .2, .8, .1, 6, .6, .3, 11, .6], [3, 3, 1]);
        var input = tf.concat([input1, input2, input3], 2);
        var result = argmax2d(input);
        var expectedResult = tf.tensor2d([2, 1, 1, 0, 2, 1], [3, 2], 'int32');
        tf.test_util.expectArraysClose(result, expectedResult);
    });
});
//# sourceMappingURL=argmax2d_test.js.map