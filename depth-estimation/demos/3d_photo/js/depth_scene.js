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
let scene, camera, uniforms;
let isMouseDown = false, mousePos = {x: 0, y: 0, z: 0, w: 0};
let start_time = new Date() / 1000;
let canvas_texture = null;
let canvas_texture2 = null;
let SCREEN_WIDTH = 192 * 2, SCREEN_HEIGHT = 256;

document.onmousedown = function() {
  isMouseDown = true;
  mousePos.z = 1;
  mousePos.w = 1;
};
document.onmouseup = function() {
  isMouseDown = false;
  mousePos.z = 0;
  mousePos.w = 0;
};

let channel0, channel1, channel2, channel3;

function initGL() {
  scene = new THREE.Scene();
  let VIEW_ANGLE = 45;  // Camera frustum vertical field of view, from bottom to
                        // top of view, in degrees.
  let ASPECT = SCREEN_WIDTH / SCREEN_HEIGHT;
  let NEAR = 0.1, FAR = 20000;
  camera = new THREE.PerspectiveCamera(VIEW_ANGLE, ASPECT, NEAR, FAR);

  topCamera = new THREE.OrthographicCamera(
      SCREEN_WIDTH / -4,   // Left
      SCREEN_WIDTH / 4,    // Right
      SCREEN_HEIGHT / 4,   // Top
      SCREEN_HEIGHT / -4,  // Bottom
      -5000,               // Near
      10000);              // Far -- enough to see the skybox
  topCamera.up = new THREE.Vector3(0, 0, -1);
  topCamera.lookAt(new THREE.Vector3(0, -1, 0));
  scene.add(topCamera);

  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setClearColor(0x000000);
  renderer.setSize(SCREEN_WIDTH, SCREEN_HEIGHT);
  container = document.getElementById('GL');
  container.appendChild(renderer.domElement);

  // depth
  canvas_texture = new THREE.CanvasTexture(document.getElementById('result'));
  // rgb
  canvas_texture2 = new THREE.CanvasTexture(document.getElementById('im1'));

  channel0 = canvas_texture;
  // FLOOR
  uniforms = {
    iChannel0: {type: 't', value: canvas_texture},
    iChannel1: {type: 't', value: canvas_texture2},
    iResolution: {type: 'v3', value: new THREE.Vector3()},
    iChannelResolution0: {
      type: 'v3',
      value: new THREE.Vector3(SCREEN_WIDTH, SCREEN_HEIGHT, 0.0)
    },
    iMouse: {type: 'v4', value: new THREE.Vector4()}
  };

  let floorMaterial = new THREE.ShaderMaterial({
    uniforms: uniforms,
    overdraw: true,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    side: THREE.DoubleSide,
    transparent: false,
    glslVersion: THREE.GLSL3,
  });

  let floorGeometry =
      new THREE.PlaneGeometry(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 10, 10);
  let floor = new THREE.Mesh(floorGeometry, floorMaterial);
  floor.position.y = 0;
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
}

function animate() {
  requestAnimationFrame(animate);

  // update all uniforms
  uniforms.iResolution.value =
      new THREE.Vector3(SCREEN_WIDTH, SCREEN_HEIGHT, 0);

  uniforms.iMouse.value =
      new THREE.Vector4(mousePos.x, mousePos.y, mousePos.z, mousePos.w);
  this.renderer.render(scene, topCamera);
}

function onWindowResize() {
  topCamera.aspect = SCREEN_WIDTH / SCREEN_HEIGHT;
  topCamera.updateProjectionMatrix();

  renderer.setSize(SCREEN_WIDTH, SCREEN_HEIGHT);
};
