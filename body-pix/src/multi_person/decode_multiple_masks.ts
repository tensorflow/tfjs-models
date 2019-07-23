

import {PersonSegmentation, Pose} from '../types';

declare type Pair = {
  x: number,
  y: number,
};

const NUM_KPT_TO_USE = 17;

function computeDistance(embedding: Pair[], pose: Pose, minPartScore = 0.3) {
  let distance = 0.0;
  let numKpt = 0;
  // for (let p = 0; p < embedding.length; p++) {
  for (let p = 0; p < NUM_KPT_TO_USE; p++) {
    if (pose.keypoints[p].score > minPartScore) {
      numKpt += 1;
      distance += (embedding[p].x - pose.keypoints[p].position.x) ** 2 +
          (embedding[p].y - pose.keypoints[p].position.y) ** 2;
    }
    pose.keypoints
  }
  if (numKpt === 0) {
    distance = 100000000;
  } else {
    distance = distance / numKpt;
  }
  return distance;
}

export function decodeMultipleMasks(
    segmentation: Uint8Array, longOffsets: Float32Array, poses: Pose[],
    height: number, width: number): PersonSegmentation {
  let data = new Uint8Array(height * width);
  debugger
  for (let i = 0; i < height; i += 1) {
    for (let j = 0; j < width; j += 1) {
      const n = i * width + j;
      const prob = segmentation[n];
      if (prob > 0.5) {
        // Method I
        // 1) finds the type of keypoint (e.g. elbow) that pixel is
        // corresponding to. 2) finds the long offset (l) that corresponding to
        // the attractor (e.g. elbow) 3) loops over poses and find the instance
        // k whose `elbow` position
        //    is closest to ((i, j) + l) and assign k to the pixel (i, j).

        // Method II (currently used)
        // 1) finds the pixel's embedding vector for all keypoints
        // 2) loops over the poses and find the instnace k that is close to the
        //    embedding at the pixel and assign k to the pixel (i, j).
        let embed = [];
        for (let p = 0; p < NUM_KPT_TO_USE; p++) {
          let dy = longOffsets[17 * (2 * n) + p];
          let dx = longOffsets[17 * (2 * n + 1) + p];
          let y = i + dy;
          let x = j + dx;
          //   // embedding refinement steps.
          //   for (let t = 0; t < 2; t++) {
          //     let nn = Math.round(y) * width + Math.round(x);
          //     dy = longOffsets[17 * (2 * nn) + p];
          //     dx = longOffsets[17 * (2 * nn + 1) + p];
          //     y = y + dy;
          //     x = x + dx;
          //   }
          embed.push({y: y, x: x});
        }

        let k_min = -1;
        let k_min_dist = 1000000000.0;
        for (let k = 0; k < poses.length; k++) {
          if (poses[k].score > 0.2) {
            const dist = computeDistance(embed, poses[k]);
            if (i % 32 == 0 && j % 32 == 0) {
              console.log(`height, width (${height}, ${width})`);
              console.log(`(${i}, ${j})`);
              console.log(`(${i / 32}, ${j / 32})`);
              console.log(`pose ${k}`);
              console.log(`distance ${dist}`);
            }
            if (dist < k_min_dist) {
              k_min = k;
              k_min_dist = dist;
            }
          }
        }

        // console.log(`embd: (${embed[0].x}, ${embed[0].y})`);
        // console.log(`nose: (${poses[0].keypoints[0].position.x}, ${
        //     poses[0].keypoints[0].position.y})`);
        data[n] = k_min + 1;
        if (i % 32 == 0 && j % 32 == 0) {
          console.log(`Data: ${data[n]}`);
        }
        // console.log(k_min);
        // data[n] = 1;
      } else {
        data[n] = 0;
      }
    }
  }
  return {data: data, height: height, width: width};
}