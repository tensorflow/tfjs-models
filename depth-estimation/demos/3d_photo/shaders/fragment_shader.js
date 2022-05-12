const FRAGMENT_SHADER = `
precision highp float;

#define uResolution iResolution.xy
#define uTouch (iMouse.zwz / iResolution.xyx)
#define uDepth iChannel0
#define uColor iChannel1

uniform sampler2D iChannel0, iChannel1; //, iTexture1, iTexture2, iTexture3;
uniform vec3 iResolution; // viewport resolution (in pixels)
uniform vec4 iMouse; // mouse pixel coords. xy: current (if MLB down), zw: click
varying vec2 fragCoord; // fragCoord is the actual pixel coordinate.

out vec4 fragColor;

const float kDepthInitScale = 1.0;
const float kDepthMaxScale = 1.0;

float saturate(float x) {
  return clamp( x, 0.0, 1.0 );
}

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

// Returns the depth value at the normalized coordinates uv.
float GetDepth(in vec2 uv) {
  // return texture(uDepth, uv).r;

  ivec2 UV = ivec2(uv * uResolution);
  const float kByte = 256.0;

  return (
    texelFetch(uDepth, UV, 0).r * kDepthInitScale * kByte * kByte
    + texelFetch(uDepth, UV, 0).g * kDepthInitScale * kDepthInitScale * kByte
    + texelFetch(uDepth, UV, 0).b * kDepthInitScale
    ) / kByte / kByte;
}

vec3 Render(vec2 uv) {
  // Gets the current depth value.
  float depth  = GetDepth(uv);

  if (uv.x < 0.5) {
    return TurboPlus(depth);
  } else {
    return vec3(1.0);
  }
}

void main()
{
  vec2 uv = fragCoord.xy / iResolution.xy;
  fragColor = vec4(Render(uv), 1.0);
}
`;