const VERTEX_SHADER = `
uniform vec3 iResolution; // viewport resolution (in pixels)
varying vec2 fragCoord;

void main() {
	fragCoord = vec2(uv.x * iResolution.x, uv.y * iResolution.y);
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;