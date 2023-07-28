import { makeRollupConfig } from '../tools/make_rollup_config.mjs';

export default makeRollupConfig({
  name: 'body-pix',
  globals: {
    '@mediapipe/selfie_segmentation': 'globalThis'
  }
});
