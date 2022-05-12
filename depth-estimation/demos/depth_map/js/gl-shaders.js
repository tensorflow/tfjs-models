/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const PASSTHROUGH_VERTEX_SHADER = `#version 300 es
precision highp float;
in vec4 position;
in vec4 input_tex_coord;

out vec2 tex_coord;

void main() {
  gl_Position = position;
  tex_coord = input_tex_coord.xy;
}`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

uniform sampler2D frame;
uniform sampler2D mask;

in highp vec2 tex_coord;
out vec4 out_color;

#define uDepth mask
#define uColor frame

#define GetDepth(uv) (texture(uDepth, uv).r)
#define GetColor(uv) (texture(uColor, uv).rgb)

float saturate(float x) {
  return clamp(x, 0.0, 1.0);
}

uniform float kVisualizeDepth;
vec3 TurboPlus(in float x) {
  const float d = 1. / 32.;
  // if input x is in meters rather than [0-1]: uncomment the following line:
  // if (x < 8f) { x /= 8f; } else { x = (x - 8f) / 24f }
  const vec3[] kColors = vec3[](
    vec3(0.0, 0.0, 0.0), vec3(0.6754, 0.0898, 0.0045), vec3(0.8240, 0.1918, 0.0197), vec3(0.9262, 0.3247, 0.0584), vec3(0.9859, 0.5048, 0.1337), vec3(0.9916, 0.6841, 0.2071), vec3(0.9267, 0.8203, 0.2257), vec3(0.7952, 0.9303, 0.2039), vec3(0.6332, 0.9919, 0.2394), vec3(0.4123, 0.9927, 0.3983), vec3(0.1849, 0.9448, 0.6071), vec3(0.0929, 0.8588, 0.7724), vec3(0.1653, 0.7262, 0.9316), vec3(0.2625, 0.5697, 0.9977), vec3(0.337, 0.443, 0.925), vec3(0.365, 0.306, 0.859),
    vec3(0.4310, 0.1800, 0.827),  vec3(0.576, 0.118, 0.859), vec3(0.737, 0.200, 0.886), vec3(0.8947, 0.2510, 0.9137), vec3(1.0000, 0.3804, 0.8431), vec3(1.0000, 0.4902, 0.7451), vec3(1.0000, 0.5961, 0.6471), vec3(1.0000, 0.6902, 0.6039), vec3(1.0000, 0.7333, 0.6157), vec3(1.0000, 0.7804, 0.6431), vec3(1.0000, 0.8275, 0.6824), vec3(1.0000, 0.8706, 0.7255), vec3(1.0000, 0.9098, 0.7765), vec3(1.0000, 0.9451, 0.8235), vec3(1.0000, 0.9725, 0.8588), vec3(1.0000, 0.9922, 0.8863),
    vec3(1., 1., 1.)
  );

  vec3 col = vec3(0.0);
  for (float i = 0.; i < 32.; i += 1.) {
    col += (step(d*i, x) - step(d *(i+1.), x)) * mix(kColors[int(i)], kColors[int(i+1.)], saturate((x-d*i)/d));
  }

  // Adds the last white colors after 99%.
  col += step(.99, x) * mix(kColors[31], kColors[32], saturate((x-.99)/.01));

  return col;
}

// Relights a scene with distance in 3D, see go/motionlights-doc.
vec3 RenderMotionLights(in vec2 uv) {
  vec3 color = GetColor(uv);
  float depth = GetDepth(uv);
  if (kVisualizeDepth > 0.5f) {
    const float uAlpha = 0.5;
    return mix(TurboPlus(depth), color, uAlpha);
  }
  return color;
}

void main() {
  // The user-facing camera is mirrored, flip horizontally.
  vec2 uv = vec2(1.0 - tex_coord[0], tex_coord[1]);
  vec3 rgb = RenderMotionLights(uv);
  out_color = vec4(rgb, 1.0);
}`;