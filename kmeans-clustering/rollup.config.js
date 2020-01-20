import node from 'rollup-plugin-node-resolve';
import typescript from 'rollup-plugin-typescript2';
import uglify from 'rollup-plugin-uglify';

const PREAMBLE =
  `// @tensorflow/tfjs-models Copyright ${(new Date).getFullYear()} Google`;

function minify() {
  return uglify({
    output: {
      preamble: PREAMBLE
    }
  });
}

function config({
  plugins = [],
  output = {}
}) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfigOverride: {
          compilerOptions: {
            module: 'ES2015'
          }
        }
      }),
      node(), ...plugins
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        '@tensorflow/tfjs': 'tf'
      },
      ...output
    },
    external: ['@tensorflow/tfjs']
  };
}

export default [
  config({
    output: {
      format: 'umd',
      name: 'kmeans',
      file: 'dist/kmeans.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: 'kmeans',
      file: 'dist/kmeans.min.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'es',
      file: 'dist/kmeans.esm.js'
    }
  })
];
