const FRAGMENT_SHADER_3D_PHOTO = `
precision highp float;

uniform sampler2D iChannel0, iChannel1;
uniform mat4 uTextureProjectionMatrix;
uniform vec3 iResolution;
uniform vec4 iMouse;
varying vec2 fragCoord;
varying vec4 mvPosition;

out vec4 fragColor;

void main()
{
  vec2 uv = fragCoord.xy / iResolution.xy;
  fragColor = vec4(texture(iChannel1, uv).rgb, 1.0);
}
`;