import * as tf from "@tensorflow/tfjs";
import {
  describeWithFlags,
  NODE_ENVS
} from "@tensorflow/tfjs-core/dist/jasmine_util";
import DummyModel from ".";

describeWithFlags("ObjectDetection", NODE_ENVS, () => {
  it("DummyModel detect method should generate no output", async () => {
    const dummy = new DummyModel();
    const x = tf.zeros([227, 227, 3]) as tf.Tensor3D;

    const data = await dummy.predict(x);

    expect(data).toEqual();
  });
});
