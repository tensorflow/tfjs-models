import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';

export default {
  input: 'index.ts',
  plugins: [
    typescript(),
    node()
  ],
  external: [
    '@tensorflow/tfjs'
  ],
  output: {
    banner: `// @tensorflow/tfjs-models Copyright ${(new Date).getFullYear()} Google`,
    file: 'dist/mobilenet.js',
    format: 'umd',
    name: 'mobilenet',
    globals: {
      '@tensorflow/tfjs': 'tf'
    }
  }
};
