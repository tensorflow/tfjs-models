/**
 * @license
 * Copyright 2023 Google LLC.
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

import '@tensorflow/tfjs-backend-webgl';
import * as tfwebgpu from '@tensorflow/tfjs-backend-webgpu';

import * as tf from '@tensorflow/tfjs-core';

async function getDevice(canvas) {
  const device = tf.backend().device;
  if (!tf.backend() instanceof tfwebgpu.WebGPUBackend) {
    throw new Error('This is only supported in WebGPU backend!');
  }
  const swapChain = canvas.getContext('webgpu');

  swapChain.configure({
    device,
    format: navigator.gpu.getPreferredCanvasFormat(),
    alphaMode: 'opaque',
  });
  return [device, swapChain];
}

function byteSizeFromShape(shape) {
  if (shape.length === 0) {
    // Scalar.
    return 4;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size * 4;
}

export class RendererWebGPU {
  constructor(device, swapChain) {
    this.device = device;
    this.swapChain = swapChain;
    this.indexBuffer = null;
    this.uniformBuffer = null;
    this.posePipeline = null;
    this.poseIndexCount = 0;
    this.texturePipeline = null;
    this.canvasInfo = null;
    this.importVideo = true;
  }

  static async setup(canvas) {
    const [device, swapChain] = await getDevice(canvas);
    return new RendererWebGPU(device, swapChain);
  }

  createBuffer(usage, size, array = null) {
    const mappedAtCreation = array ? true : false;
    const buffer = this.device.createBuffer({size, usage, mappedAtCreation});
    if (array instanceof Float32Array) {
      new Float32Array(buffer.getMappedRange()).set(array);
      buffer.unmap();
    } else if (array instanceof Uint32Array) {
      new Uint32Array(buffer.getMappedRange()).set(array);
      buffer.unmap();
    }
    return buffer;
  }

  draw(video, tensor, canvasInfo) {
    this.canvasInfo = canvasInfo;
    const videoCommanderBuffer = this.drawTexture(video);
    const poseCommanderBuffer = this.drawPose(tensor);
    this.device.queue.submit([videoCommanderBuffer, poseCommanderBuffer]);
  }

  getPoseShader(tensorShapeX) {
    const vertexShaderCode = `
struct Uniforms {
  offsetX : f32,
  offsetY : f32,
  scaleX : f32,
  scaleY : f32,
  width : f32,
  height : f32,
}

struct Pose {
  keypoints : array<vec2<f32>, ${tensorShapeX}>
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<storage> pose : Pose;
@vertex
fn main(
  @builtin(vertex_index) VertexIndex : u32
) -> @builtin(position) vec4<f32> {
  let rawY = (pose.keypoints[VertexIndex].x + uniforms.offsetY) * uniforms.scaleY / uniforms.height;
  let rawX  = (pose.keypoints[VertexIndex].y + uniforms.offsetX) * uniforms.scaleX / uniforms.width;
  var x = rawX * 2.0 - 1.0;
  var y = 1.0 - rawY * 2.0;
  return vec4<f32>(x, y, 1.0, 1.0);
}
    `;
    const fragmentShaderCode = `
@fragment
fn main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
    `;
    return [vertexShaderCode, fragmentShaderCode];
  }

  initDrawPose(tensor) {
    // Only 2d tensor whose last dimension is 2 is supported.
    if (tensor == null || tensor.shape.length !== 2 || tensor.shape[1] !== 2) {
      throw new Error('Tensor is null or tensor shape is not supported!');
    }

    // pose-detection supports 17 body parts:
    // 'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    // 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    // 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    // 'right_knee', 'left_ankle', 'right_ankle'. This demo draws the first
    // five.
    const poseIndexArray = new Uint32Array([3, 1, 1, 0, 0, 2, 2, 4]);
    this.poseIndexCount = poseIndexArray.length;

    if (this.indexBuffer == null) {
      this.indexBuffer = this.createBuffer(
          GPUBufferUsage.INDEX, poseIndexArray.byteLength, poseIndexArray);
    }
    if (this.uniformBuffer == null) {
      this.uniformBuffer = this.createBuffer(
          GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          this.canvasInfo.length * 4);
    }
    if (this.posePipeline == null) {
      this.posePipeline = this.createPosePipeline(tensor.shape[0]);
    }
    const bindings = [
      {
        buffer: this.uniformBuffer,
        offset: 0,
        size: this.canvasInfo.length * 4,
      },
      {
        buffer: tensor.dataToGPU().buffer,
        offset: 0,
        size: byteSizeFromShape(tensor.shape),
      }
    ];
    return this.device.createBindGroup({
      layout: this.posePipeline.getBindGroupLayout(0),
      entries: bindings.map((b, i) => ({binding: i, resource: b})),
    });
  }

  drawPose(tensor) {
    const poseBindGroup = this.initDrawPose(tensor);
    const textureView = this.swapChain.getCurrentTexture().createView();

    const uniformData = new Float32Array(this.canvasInfo);
    this.device.queue.writeBuffer(
        this.uniformBuffer, 0, uniformData.buffer, uniformData.byteOffset,
        uniformData.byteLength);

    const renderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        loadValue: {r: 0.5, g: 0.5, b: 0.5, a: 1.0},
        loadOp: 'load',
        storeOp: 'store',
      }]
    };
    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.posePipeline);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.setBindGroup(0, poseBindGroup);
    passEncoder.drawIndexed(this.poseIndexCount);
    passEncoder.end();
    return commandEncoder.finish();
  }

  createPosePipeline(tensorShapeX) {
    const [vertexShaderCode, fragmentShaderCode] =
        this.getPoseShader(tensorShapeX);
    return this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: this.device.createShaderModule({code: vertexShaderCode}),
        entryPoint: 'main',
      },
      fragment: {
        module: this.device.createShaderModule({code: fragmentShaderCode}),
        entryPoint: 'main',
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: {
        topology: 'line-list',
      }
    });
  }

  getExternalTextureShader() {
    const vertexShaderCode = `
@vertex fn main(@builtin(vertex_index) VertexIndex : u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec4<f32>, 6>(
    vec4<f32>( 1.0, 1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0, 1.0, 0.0, 1.0),
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0, 1.0, 0.0, 1.0)
  );
  return pos[VertexIndex];
}
      `;
    const textureType =
        this.importVideo ? 'texture_external' : 'texture_2d<f32>';
    const fragmentShaderCode = `
@group(0) @binding(0) var s : sampler;
@group(0) @binding(1) var t : ${textureType};

@fragment fn main(@builtin(position) FragCoord : vec4<f32>)
                         -> @location(0) vec4<f32> {
    return textureSampleBaseClampToEdge(t, s, FragCoord.xy / vec2<f32>(${
        this.canvasInfo[4]}, ${this.canvasInfo[5]}));
}
      `;
    return [vertexShaderCode, fragmentShaderCode];
  }

  drawTexture(video) {
    const textureBindGroup = this.initDrawTexture(video);
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.swapChain.getCurrentTexture().createView();

    const renderPassDescriptor = {
      colorAttachments: [{
        view: textureView,
        loadValue: {r: 0.5, g: 0.5, b: 0.5, a: 1.0},
        loadOp: 'clear',
        storeOp: 'store',
      }]
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.setPipeline(this.texturePipeline);
    passEncoder.setBindGroup(0, textureBindGroup);
    passEncoder.draw(6);
    passEncoder.end();
    return commandEncoder.finish();
  }

  createTexturePipeline() {
    const [vertexShaderCode, fragmentShaderCode] =
        this.getExternalTextureShader();
    this.texturePipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: this.device.createShaderModule({
          code: vertexShaderCode,
        }),
        entryPoint: 'main',
      },
      fragment: {
        module: this.device.createShaderModule({
          code: fragmentShaderCode,
        }),
        entryPoint: 'main',
        targets: [
          {
            format: 'bgra8unorm',
          },
        ],
      },
      primitive: {topology: 'triangle-list'},
    });
  }

  initDrawTexture(video) {
    if (this.texturePipeline == null) {
      this.createTexturePipeline();
    }

    const linearSampler = this.device.createSampler();
    let externalTexture;
    if (this.importVideo) {
      externalTexture = this.device.importExternalTexture({
        source: video,
      });
    } else {
      const width = this.canvasInfo[4];
      const height = this.canvasInfo[5];
      const format = 'rgba8unorm';
      const usage = GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
      externalTexture = this.device.createTexture({
        size: [width, height],
        format,
        usage,
      });
      this.device.queue.copyExternalImageToTexture(
          {source: video}, {texture: externalTexture}, [width, height]);
    }

    const bindGroup = this.device.createBindGroup({
      layout: this.texturePipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: linearSampler,
        },
        {
          binding: 1,
          resource: this.importVideo ? externalTexture :
                                       externalTexture.createView(),
        },
      ],
    });

    return bindGroup;
  }

  dispose() {
    this.indexBuffer.destroy();
    this.uniformBuffer.destroy();
  }
}