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
import {GUI} from 'https://cdn.jsdelivr.net/npm/lil-gui@0.16.1/dist/lil-gui.esm.min.js';
import * as THREE from 'three';

let depthData;
let camera, scene, renderer;
let depthMaterial;
let mesh;
let backgroundMesh;
let INVALID_DEPTH_THRES = 0.1;
let INVALID_DEPTH = 10000;
let RENDERER_WIDTH = 192, RENDERER_HEIGHT = 256;

const geometry = new THREE.BufferGeometry();

let IMAGE_WIDTH = 192, IMAGE_HEIGHT = 256;
const vertices = [];

setTimeout(function() {
  init();
  animate();
}, 1000);

function getDepth(depthData, vid) {
  // vid: vertex id.
  const depth0 = depthData[vid * 4 + 0];
  const depth1 = depthData[vid * 4 + 1];
  const depth2 = depthData[vid * 4 + 2];
  let depth = depth0 * 255 * 255 + depth1 * 255 + depth2;
  depth = depth / 255 / 255 / 255;
  if (isNaN(depth)) {
    depth = 0;
  }
  if (depth <= INVALID_DEPTH_THRES) depth = INVALID_DEPTH;
  return depth;
}

function getIndices(depthData) {
  let indices = [];
  for (let i = 0; i < IMAGE_HEIGHT; i++) {
    for (let j = 0; j < IMAGE_WIDTH; j++) {
      const a = i * (IMAGE_WIDTH + 1) + (j + 1);
      const b = i * (IMAGE_WIDTH + 1) + j;
      const c = (i + 1) * (IMAGE_WIDTH + 1) + j;
      const d = (i + 1) * (IMAGE_WIDTH + 1) + (j + 1);

      let aDepth = getDepth(depthData, i * IMAGE_WIDTH + j + 1);
      let bDepth = getDepth(depthData, i * IMAGE_WIDTH + j);
      let cDepth = getDepth(depthData, (i + 1) * IMAGE_WIDTH + j);
      let dDepth = getDepth(depthData, (i + 1) * IMAGE_WIDTH + j + 1);
      // generate two faces (triangles) per iteration

      if (aDepth != INVALID_DEPTH && bDepth != INVALID_DEPTH &&
          dDepth != INVALID_DEPTH) {
        indices.push(a, b, d);  // face one
      }

      if (bDepth != INVALID_DEPTH && cDepth != INVALID_DEPTH &&
          dDepth != INVALID_DEPTH) {
        indices.push(b, c, d);  // face two
      }
    }
  }

  return indices;
}

updateDepthCallback = () => {
  depthData = document.getElementById('result')
                  .getContext('2d')
                  .getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
                  .data;

  for (let i = 0; i <= IMAGE_HEIGHT; ++i) {
    const y = i - IMAGE_HEIGHT * 0.5;
    for (let j = 0; j <= IMAGE_WIDTH; ++j) {
      const x = j - IMAGE_WIDTH * 0.5;
      const vid = i * IMAGE_WIDTH + j;
      let depth = getDepth(depthData, vid);
      const vid2 = i * (IMAGE_WIDTH + 1) + j;
      vertices[vid2 * 3 + 2] = depth * config.depthScale;
    }
  }

  const indices = getIndices(depthData);

  //
  geometry.setIndex(indices);
  geometry.setAttribute(
      'position', new THREE.Float32BufferAttribute(vertices, 3));

  depthMaterial.uniforms.iChannel0.value.needsUpdate = true;
  depthMaterial.uniforms.iChannel1.value.needsUpdate = true;
  geometry.attributes.position.needsUpdate = true;
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
};

function startRecording(format) {
  if (!config.autoAnimation) {
    alert('autoAnimation must be turned on!');
    return;
  }
  if (capturer) {
    capturer.stop();
  }
  capturer = new CCapture({
    format,
    name: 'portrait',
    verbose: false,
    workersPath: './js/',
  });

  capturer.start();
  capturerInitialTheta = Date.now() * config.cameraSpeed;
}

document.getElementById('download-anim-gif').onclick = () =>
    startRecording('gif');
document.getElementById('download-anim-webm').onclick = () =>
    startRecording('webm');

function init() {
  const FOV = 27;
  camera = new THREE.PerspectiveCamera(
      FOV, RENDERER_WIDTH / RENDERER_HEIGHT, 0.001, 3500);
  camera.position.z = 7;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(config.backgroundColor);

  const normals = [];
  const colors = [];
  const uvs = [];

  let depth_texture =
      new THREE.CanvasTexture(document.getElementById('result'));
  let image_texture = new THREE.CanvasTexture(document.getElementById('im1'));

  depthData = document.getElementById('result')
                  .getContext('2d')
                  .getImageData(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
                  .data;

  for (let i = 0; i <= IMAGE_HEIGHT; ++i) {
    const y = i - IMAGE_HEIGHT * 0.5;
    for (let j = 0; j <= IMAGE_WIDTH; ++j) {
      const x = j - IMAGE_WIDTH * 0.5;

      const vid = i * IMAGE_WIDTH + j;
      let depth = getDepth(depthData, vid);

      vertices.push(
          x * config.imageScale, -y * config.imageScale,
          depth * config.depthScale);
      normals.push(0, 0, 1);

      const r = (x / IMAGE_WIDTH) + 0.5;
      const g = (y / IMAGE_HEIGHT) + 0.5;
      colors.push(r, g, 1);

      uvs.push(j / IMAGE_WIDTH, 1.0 - i / IMAGE_HEIGHT);
    }
  }

  const indices = getIndices(depthData);

  //
  geometry.setIndex(indices);
  geometry.setAttribute(
      'position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

  camera.aspect = RENDERER_WIDTH / RENDERER_HEIGHT;
  camera.updateProjectionMatrix();

  let uniforms = {
    iChannel0: {type: 't', value: depth_texture},
    iChannel1: {type: 't', value: image_texture},
    iResolution:
        {type: 'v3', value: new THREE.Vector3(IMAGE_WIDTH, IMAGE_HEIGHT, 0)},
    iChannelResolution0:
        {type: 'v3', value: new THREE.Vector3(512.0 * 2, 512.0 * 2, 0.0)},
    iMouse: {type: 'v4', value: new THREE.Vector4()},
    uTextureProjectionMatrix: {type: 'm4', value: camera.projectionMatrix}
  };

  depthMaterial = new THREE.ShaderMaterial({
    uniforms: uniforms,
    overdraw: true,
    vertexShader: VERTEX_SHADER_3D_PHOTO,
    fragmentShader: FRAGMENT_SHADER_3D_PHOTO,
    transparent: false,
    wireframe: false,
    wireframeLinewidth: 2,
    glslVersion: THREE.GLSL3,
  });

  const PLANE_SIZE = 0.025;
  let planeGeometry = new THREE.PlaneGeometry(
      IMAGE_WIDTH * PLANE_SIZE, IMAGE_HEIGHT * PLANE_SIZE, 10, 10);

  mesh = new THREE.Mesh(geometry, depthMaterial);
  backgroundMesh = new THREE.Mesh(planeGeometry, depthMaterial);
  scene.add(mesh);
  scene.add(backgroundMesh);

  renderer =
      // Ensure buffer is preserved for CCapture.js.
      new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(RENDERER_WIDTH, RENDERER_HEIGHT);

  let container = document.getElementById('GL2');
  container.appendChild(renderer.domElement);

  const gui = new GUI();

  let vis = gui.addFolder('Visualization');
  vis.add(depthMaterial, 'wireframe');
  const minDepthController = vis.add(config, 'minDepth', 0, 1);
  minDepthController.onChange(predict);
  const maxDepthController = vis.add(config, 'maxDepth', 0, 1);
  maxDepthController.onChange(predict);
  vis.closed = false;

  let bkg = gui.addFolder('background');
  bkg.addColor(config, 'backgroundColor');
  bkg.add(config, 'showBackgroundPic');

  let cam = gui.addFolder('camera');
  cam.add(config, 'autoAnimation');
  cam.add(config, 'cameraSpeed', 0, 0.01);
  cam.add(config, 'cameraRadius', 0, 1);
  cam.add(config, 'cameraZ', -20, 20);

  gui.close();
}

function animate() {
  requestAnimationFrame(animate);
  render();
}

function render() {
  const time = Date.now() * 0.001;
  scene.background = new THREE.Color(config.backgroundColor);
  backgroundMesh.position.set(0, 0, config.backgroundDepth);
  backgroundMesh.scale.set(
      config.backgroundScale, config.backgroundScale, config.backgroundScale);
  backgroundMesh.visible = config.showBackgroundPic;

  if (config.autoAnimation) {
    const _cameraRotationSpeed = config.cameraSpeed;
    const _cameraRotationRadius = config.cameraRadius;
    let theta = Date.now() * _cameraRotationSpeed;
    let cachedCameraPosition =
        new THREE.Vector3(config.cameraX, config.cameraY, config.cameraZ);
    let dirA = (new THREE.Vector3(1, 1, 0)).multiplyScalar(Math.cos(theta));
    let dirB = (new THREE.Vector3(0, 1, 1)).multiplyScalar(Math.sin(theta));
    let dir = dirA.add(dirB).multiplyScalar(_cameraRotationRadius);
    let newPosition = cachedCameraPosition.add(dir);
    camera.position.set(newPosition.x, newPosition.y, newPosition.z);
    camera.lookAt(
        config.cameraLookAtX, config.cameraLookAtY, config.cameraLookAtZ);

    if (capturer != null) {
      const capturerCanvas = document.createElement('canvas');
      capturerCanvas.width = RENDERER_WIDTH;
      capturerCanvas.height = RENDERER_HEIGHT;
      const capturerContext = capturerCanvas.getContext('2d');
      capturerContext.drawImage(
          renderer.domElement, 0, 0, RENDERER_WIDTH, RENDERER_HEIGHT);
      capturer.capture(capturerCanvas);
      if (theta >= capturerInitialTheta + 2 * Math.PI) {
        capturer.save();
        capturer.stop();
        capturer = null;
        capturerInitialTheta = null;
      }
    }
  }

  renderer.render(scene, camera);
}
