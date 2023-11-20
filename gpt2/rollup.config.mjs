import typescript from "rollup-plugin-typescript2";
import json from '@rollup/plugin-json';

export default {
  input: './src/gpt2.ts',
  plugins: [
    typescript({
      tsconfig: "./tsconfig.json",
    }),
    json()
  ]
};
