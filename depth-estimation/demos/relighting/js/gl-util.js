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

function getWebGLRenderingContext(canvas) {
    const WEBGL_ATTRIBUTES = {
      alpha: true,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true,
      powerPreference: 'low-power'
    };
    return canvas.getContext('webgl2', WEBGL_ATTRIBUTES);
}

const checkGlError = (gl, label, allowFail = false) => {
    let err = gl.getError();
    let foundError = false;
  
    while (err != gl.NO_ERROR) {
      let error;
  
      switch (err) {
        case gl.INVALID_OPERATION:
          error = 'INVALID_OPERATION';
          break;
        case gl.INVALID_ENUM:
          error = 'INVALID_ENUM';
          break;
        case gl.INVALID_VALUE:
          error = 'INVALID_VALUE';
          break;
        case gl.OUT_OF_MEMORY:
          error = 'OUT_OF_MEMORY';
          break;
        case gl.INVALID_FRAMEBUFFER_OPERATION:
          error = 'INVALID_FRAMEBUFFER_OPERATION';
          break;
      }
      // When debugging or running tests, this is a fatal, non-recoverable
      // error. Otherwise program can resume operation.
      if (!allowFail) {
        throw new Error(`GL error: ${error} ${label}`);
      }
      log.error(logger, `GL error: GL_${error}: ${label}`);
      err = gl.getError();
  
      foundError = true;
    }
    return foundError;
};

const compileShader = (gl, type, src) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    const logMsg = gl.getShaderInfoLog(shader);
    if (logMsg) {
      console.warn(logMsg);
    }
    return shader;
};
  
// Creates a WebGLTexture from pixel data.
const createWebGLTexture =
    (gl, internalFormat, format, type, filterMode, pixelData, width,
    height) => {
    checkGlError(gl, 'Before Texture::Create');

    const texture = gl.createTexture();

    // Set up the texture.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filterMode);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filterMode);

    // Create (and load if needed) texture.
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(
        gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type,
        pixelData);

    checkGlError(gl, 'Texture::Create');
    return texture;
};

// Returns a wrapper class with a framebuffer and output texture for it.
function createTextureFrameBuffer(gl, filterMode, width, height) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);

    const texture = createWebGLTexture(
        gl, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, filterMode, null, width,
        height);
    checkGlError(gl, 'TextureFramebuffer::Create: create texture');

    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);

    if (status != gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Bad framebuffer status:${status}`);
    }

    return new GlTextureFramebuffer(gl, framebuffer, texture, width, height);
  }

  // Returns a wrapper class with a texture and its util.
  function createTexture(gl, texture, width, height) {
    return new GlTextureImpl(gl, texture, width, height);
  }

  // Returns a wrapper class with a WebGL program and its util functions.
  function createProgram(gl, vertexSrc, fragmentSrc) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSrc);
    checkGlError(gl, 'Compile vertex shader');

    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSrc);
    checkGlError(gl, 'Compile fragment shader');

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);

    gl.bindAttribLocation(program, 0, 'position');
    gl.bindAttribLocation(program, 1, 'input_tex_coord');
    gl.linkProgram(program);
    gl.useProgram(program);

    checkGlError(gl, 'createProgram');

    return new GlProgramImpl(gl, program);
  }