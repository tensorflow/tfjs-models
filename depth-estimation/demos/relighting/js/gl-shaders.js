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

const COMMON = `#version 300 es
precision mediump float;

uniform vec2 iMouse;
uniform vec2 iResolution;
uniform float iTime;
uniform sampler2D frame;
uniform sampler2D mask;

in highp vec2 tex_coord;
out vec4 out_color;

#define uResolution iResolution.xy
#define uTouch (iMouse.xy / iResolution.xy)
#define uDepth mask
#define uColor frame

#define GetDepth(uv) (texture(uDepth, uv).r)
#define GetColor(uv) (texture(uColor, uv).rgb)

// Computes the aspect ratio for portait and landscape modes.
vec2 CalculateAspectRatio(in vec2 size) {
  return pow(size.yy / size, vec2(step(size.x, size.y) * 2.0 - 1.0));
}

// Normalizes a coordinate system from [0, 1] to [-1, 1].
vec2 NormalizeCoord(in vec2 coord, in vec2 aspect_ratio) {
  return (2.0 * coord - vec2(1.0)) * aspect_ratio;
}
// Reverts a coordinate system from [-1, 1] to [0, 1].
vec2 ReverseNormalizeCoord(in vec2 pos, in vec2 aspect_ratio) {
  return (pos / aspect_ratio + 1.0) * 0.5;
}

float saturate(float x) {
  return clamp(x, 0.0, 1.0);
}

uniform float kVisualizeDepth;
vec3 TurboPlus(in float x) {
  const float d = 1. / 32.;
  // if input x is in meters rather than [0-1]: uncomment the following line:
  // if (x < 8f) { x /= 8f; } else { x = (x - 8f) / 24f }
  const vec3[] kColors = vec3[](
    vec3(0.4796, 0.0158, 0.0106), vec3(0.6754, 0.0898, 0.0045), vec3(0.8240, 0.1918, 0.0197), vec3(0.9262, 0.3247, 0.0584), vec3(0.9859, 0.5048, 0.1337), vec3(0.9916, 0.6841, 0.2071), vec3(0.9267, 0.8203, 0.2257), vec3(0.7952, 0.9303, 0.2039), vec3(0.6332, 0.9919, 0.2394), vec3(0.4123, 0.9927, 0.3983), vec3(0.1849, 0.9448, 0.6071), vec3(0.0929, 0.8588, 0.7724), vec3(0.1653, 0.7262, 0.9316), vec3(0.2625, 0.5697, 0.9977), vec3(0.337, 0.443, 0.925), vec3(0.365, 0.306, 0.859),
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
`;

const SUNBEAM_SHADER = `${COMMON}
const float kHash1 = 47510.5358;
const vec2 kHash2 = vec2(25.5359, 77.1053);
const vec3 kHash3 = vec3(55.9973, 65.9157, 31.3163);

// GPU Hashing: vec2 in, vec2 out.
vec2 Hash22(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * kHash3);
    p3 += dot(p3, p3.yzx + kHash2.x);
    return fract((p3.xx + p3.yz) * p3.zy);

}

const vec3 kSunlightColor = vec3(1.0, 0.8, 0.46);

uniform float kLightDepth;

uniform float kRayMarchingSteps;

uniform float kGlobalBrightness;
uniform float kLowerIntensity;
uniform float kHigherIntensity;
uniform float kMaxIntensity;
uniform float kEnergyDecayFactor;

uniform float kDepthWeight;

uniform float kLightRadius;
uniform float kLightSelfBrightness;

// Returns the distance from the screen borders to the foveal.
float GetFoveationFactor(in vec2 uv) {
  const float kFoveatationFactor = 8.0;
  float foveated_distance = uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y);
  float powered_distance = pow(kFoveatationFactor * foveated_distance, kHigherIntensity);
  return kLowerIntensity * (1.0 + powered_distance);
}

vec2 GetScatterFactorOverTime(in vec2 uv) {
  return 0.9 + 0.1 * Hash22(uv + iTime);
}

// Relights a scene with distance in 3D, see go/motionlights-doc.
vec3 RenderMotionLights(in vec2 uv) {
  float depth = GetDepth(uv);
  if (kVisualizeDepth > 0.5f) {
    return TurboPlus(depth);
  }
  // Samples:
  vec2 aspect_ratio = CalculateAspectRatio(uResolution);
  vec2 normalized_touch = NormalizeCoord(uTouch.xy, aspect_ratio);
    const float debug_depth = 0.5;
  vec3 center = vec3(normalized_touch, debug_depth);
  center.z = kLightDepth;//abs(cos(0.0005 * iTime));
  
  vec2 normalized_uv = NormalizeCoord(uv, aspect_ratio);
  vec3 pos = vec3(normalized_uv, depth);
  float intensity = 0.0;
  float sample_energy = 1.0;
  vec2 light_direction = center.xy - normalized_uv;
  vec2 sample_st = normalized_uv;
  float dist = mix(distance(pos.xy, center.xy),
                   distance(pos.zz, center.zz), kDepthWeight);
  
  for (float i = 0.0; i < kRayMarchingSteps; ++i) {
    vec2 sample_uv = ReverseNormalizeCoord(sample_st, aspect_ratio);
    float k = mix(-1.0, 1.0, center.z);
    float sample_depth = GetDepth(sample_uv);
    float deviation = sample_depth * 2.0 - 1.0;
    intensity += (k * deviation + 1.0 + center.z - sample_depth) * sample_energy * kMaxIntensity;
    
    sample_energy *= kEnergyDecayFactor;
    vec2 sample_scatter = GetScatterFactorOverTime(sample_st);
    sample_st += light_direction * sample_scatter / float(kRayMarchingSteps);
  }
  intensity /= float(kRayMarchingSteps);
    
  // Result from the first pass
  vec3 col = GetColor(uv);

  intensity -= 1.0;

  col += kGlobalBrightness * abs(0.5 - intensity * kSunlightColor) * (pow(col, vec3(1.5 - intensity * kSunlightColor)) - col);
  col = clamp(col, 0.0, 1.0);
  col *= GetFoveationFactor(uv);    
  col = smoothstep(0.0, 0.7, col + 0.05);
  col = pow(col, vec3(1.0 / 1.8));
  
  // Perceptual light radius propotional to percentage in the screen space.
  float light_radius = 2.0 * atan(kLightRadius, 2.0 * (1.0 - center.z));

  float l = distance(center.xy, normalized_uv);
  if (l < light_radius && (center.z > depth)) {
    intensity = smoothstep(1.0, 0.0, l / light_radius);
    col = mix(col, kSunlightColor * kLightSelfBrightness, intensity);
  }
  return col;
}
void main() {
  // The user-facing camera is mirrored, flip horizontally.
  vec2 uv = vec2(1.0 - tex_coord[0], tex_coord[1]);
  vec3 rgb = RenderMotionLights(uv);
  out_color = vec4(rgb, 1.0);
}`;

const POINT_LIGHTS_SHADER = `${COMMON}
uniform float kRayMarchingSteps;
uniform float kLightsOffsetX;
uniform float kLightsOffsetY;
uniform float kLightsOffsetZ;

uniform vec3 kFirstLightColor;
uniform vec3 kSecondLightColor;
uniform vec3 kThirdLightColor;

uniform float kGlobalDarkness;

const float kDepthWeight = 0.9;

// Returns the squared distance between two 2D points.
float SquaredDistance(in vec2 a, in vec2 b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

// Returns the distance field of a 2D circle with anti-aliasing.
float SmoothCircle(in vec2 uv, in vec2 origin, in float radius, in float alias) {
    return 1.0 - smoothstep(radius - alias, radius + alias,
    length(uv - origin));
}


// Returns a simplified distance field of a 3D sphere with anti-aliasing
float SmoothSphere(in vec2 uv, in vec3 origin, in float depth, in float radius,
in float delta, in float alias) {
    radius = mix(radius - delta, radius + delta, -log(origin.z));
    float lightSphericalDepth = radius * radius - SquaredDistance(origin.xy, uv) * 2.0;
    return SmoothCircle(uv, origin.xy, radius, alias) *
    smoothstep(origin.z- delta, origin.z + delta, depth + lightSphericalDepth);
}

// Relights a scene with distance in 3D, see go/motionlights-doc.
vec3 RenderMotionLights(in vec2 uv) {
  float depth = GetDepth(uv);
  if (kVisualizeDepth > 0.5f) {
    return TurboPlus(depth);
  }
  // Samples:
  vec2 aspect_ratio = CalculateAspectRatio(uResolution);
  vec2 normalized_touch = NormalizeCoord(uTouch.xy, aspect_ratio);
    const float debug_depth = 0.5;
  vec3 center = vec3(normalized_touch, debug_depth);
  center.z = -0.5;
  
  vec2 normalizedUv = NormalizeCoord(uv, aspect_ratio);
  vec3 samplePos = vec3(normalizedUv, depth);
  
  vec2 sampleUv = normalizedUv;  
  float intensitySum = 0.0;
  vec3 relightsSum = vec3(0.0);

  const int kMaxNumDirectionalLights = 3;
  vec3[] _PointLightPositions = vec3[] (
      vec3(cos(iTime * 1.2) + 1.2 - kLightsOffsetX, sin(iTime * 1.6) + 0.7 - kLightsOffsetY, cos(iTime * 1.5) + 0.5 + kLightsOffsetZ),
      vec3(cos(iTime * 1.7) + 1.0 - kLightsOffsetX, sin(iTime * 1.4) - kLightsOffsetY, sin(iTime * 1.2) + kLightsOffsetZ),
      vec3(cos(iTime * 1.8) + 0.5 - kLightsOffsetX, sin(iTime * 1.8) + 0.2 - kLightsOffsetY, cos(iTime * 1.4) + kLightsOffsetZ)
      );

   for (int i = 0; i < kMaxNumDirectionalLights; ++i) {
    _PointLightPositions[i] =  _PointLightPositions[i] * 0.3 + vec3(0.25, 0.25, 0.25);
    
   }

  vec3[] _PointLightColors = vec3[] (
      kFirstLightColor / 255.0,
      kSecondLightColor / 255.0,
      kThirdLightColor / 255.0
      );

  for (int i = 0; i < kMaxNumDirectionalLights; ++i) {
    vec3 lightColor = vec3(1.0);
    lightColor = _PointLightColors[i];

    vec2 lightUv = uTouch.xy;
    vec3 lightPos = _PointLightPositions[i];

    float lightDepth = lightPos.z;

    float sample_energy = 1.0;
    vec2 light_direction = center.xy - normalizedUv;
    vec2 sample_st = normalizedUv;

    float uvDist = distance(sampleUv, lightUv);
    float depthDist = distance(samplePos.zz, lightPos.zz);
  
    float dist = mix(uvDist, depthDist, kDepthWeight);

    vec2 sampleUv = normalizedUv;
    vec3 samplePos = vec3(sampleUv, depth);
    const float kGlobalDepthWeight = 0.6;
    float globalDist = mix(uvDist, depthDist, kGlobalDepthWeight);
    vec2 lightDirection = lightUv - sampleUv;
    float distFactor = mix(0.1, 3.0, 1.0 - clamp(globalDist * 0.5, 0.0, 1.0));
    float photon_energy = 4.0 + distFactor;
    vec2 photonUv = sampleUv;

    float intensity = 0.0;

    const float kMaxIntensity = 3.0;
    const float kEnergyDecayFactor = 0.5;
    const int kNumPasses = 8;
    const float kNumPassesFloat = 8.0f;

    for (float j = 0.0; j < kRayMarchingSteps; ++j) {
      vec2 photonDepthUv = ReverseNormalizeCoord(photonUv, aspect_ratio);
      float photonDepth = GetDepth(photonDepthUv);
      float uvDist = distance(photonUv, lightUv);
      float depthDist = distance(vec2(photonDepth, photonDepth), lightPos.zz);
      const float depthWeight = 0.8;
      float dist = mix(uvDist, depthDist, depthWeight);
      float deltaIntensity = (1.0 - dist) * (1.0 - dist) * photon_energy * kMaxIntensity;
      photon_energy *= kEnergyDecayFactor;
      photonUv += lightDirection / kNumPassesFloat;
      intensity += deltaIntensity;
    }

      intensity *= saturate(1.4 - depth * 1.2);
      intensity /= kNumPassesFloat * kMaxIntensity;
      intensity = clamp(intensity, -6.0 / 3.0, 0.85);
      intensitySum += saturate(intensity);
      relightsSum += intensity * lightColor;

  }
    
  // Result from the first pass
  vec3 col = GetColor(uv);
  vec3 result = col;
  result = mix(result, result * 0.5, kGlobalDarkness);

  result += 3.0 * abs(0.5 - relightsSum) *
          (pow(result, vec3(1.5 - relightsSum)) - result);
  
   // Renders the light sources.
  vec3 outerColor = vec3(0.0);
  vec3 innerSingle = vec3(0.0);
  float innerAlpha = 0.0;

  for (int i = 0; i < kMaxNumDirectionalLights; ++i) {
    vec2 lightUv = uTouch.xy;
    vec3 lightPos = vec3(lightUv, 0.5);
    lightPos = _PointLightPositions[i];
    lightUv = lightPos.xy;
    vec3 lightColor = _PointLightColors[i];

    vec3 lightNormalizedPos = vec3(NormalizeCoord(lightPos.xy, aspect_ratio), lightPos.z);
    float lightDepth = lightPos.z;

    if (lightDepth <= 0.0) {
      continue;
    }

    // Renders the light sources.
    const float kPointLightRadius = 0.04;
    const float kPointLightFeathering = 0.02;
    float sphereDist = SmoothSphere(normalizedUv, lightNormalizedPos, depth,
                                    kPointLightRadius, kPointLightFeathering, 0.04);

    lightColor = lightColor * 2.0;

    outerColor += mix(vec3(0.0), lightColor, sphereDist);

    lightColor = mix(lightColor, vec3(1.0), 0.9);
    float innerDist = SmoothSphere(normalizedUv, lightNormalizedPos, depth, 0.025, 0.01, 0.003);

    innerSingle += mix(vec3(0.0), lightColor, step(0.03, innerDist));
    innerAlpha = max(innerAlpha, innerDist);
  }

  // Blends in global darkness around the point lights to increase the relighting effects.
  const float kFalloff = 0.001;
  const float kShadow = 0.3;
  const float kReduction = 0.3;
  float feathering = (1.0 - saturate(intensitySum * 0.9)) * (kShadow + kFalloff);
  vec3 darkness = result * smoothstep(kReduction, kFalloff, feathering);
  result = mix(result, darkness, kGlobalDarkness);

  // Blends in with the point light sources.
  result = mix(result,  outerColor, 0.3);
  result = mix(result, innerSingle, innerAlpha);

  return result;
}
void main() {
  // The user-facing camera is mirrored, flip horizontally.
  vec2 uv = vec2(1.0 - tex_coord[0], tex_coord[1]);
  vec3 rgb = RenderMotionLights(uv);
  out_color = vec4(rgb, 1.0);
}`;