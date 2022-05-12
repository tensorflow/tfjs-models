const VERTEX_SHADER_3D_PHOTO = `
precision highp float;
uniform vec3 iResolution; // viewport resolution (in pixels)
varying vec2 fragCoord, vUv;
varying vec4 mvPosition;

void main() {
  vUv = uv;
	fragCoord = vec2(uv.x * iResolution.x, uv.y * iResolution.y);
  mvPosition = modelViewMatrix * vec4(position, 1.0);
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;