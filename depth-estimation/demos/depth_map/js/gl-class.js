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

// A wrapper class for WebGL texture and its utility functions.
class GlTextureImpl {
  constructor(gl, texture, width, height) {
    this.gl_ = gl;
    this.texture_ = texture;
    this.width = width;
    this.height = height;
  }

  bindTexture() {
    this.gl_.bindTexture(this.gl_.TEXTURE_2D, this.texture_);
  }
}

// A wrapper class for WebGL texture and its associted framebuffer and utility
// functions.
class GlTextureFramebuffer extends GlTextureImpl {
  constructor(gl, framebuffer, texture, width, height) {
    super(gl, texture, width, height);
    this.framebuffer_ = framebuffer;
  }

  bindFramebuffer() {
    this.gl_.bindFramebuffer(this.gl_.FRAMEBUFFER, this.framebuffer_);
    this.gl_.viewport(0, 0, this.width, this.height);
  }
}

// A wrapper class for WebGL program and its utility functions.
class GlProgramImpl {
  constructor(gl, program) {
    this.gl_ = gl;
    this.program_ = program;
    this.cachedUniformLocations_ = new Map();
  }

  useProgram() {
    this.gl_.useProgram(this.program_);
  }

  getUniformLocation(symbol) {
    if (this.cachedUniformLocations_.has(symbol)) {
      return this.cachedUniformLocations_.get(symbol);
    } else {
      const location = this.gl_.getUniformLocation(this.program_, symbol);
      this.cachedUniformLocations_.set(symbol, location);
      return location;
    }
  }
}

// Utility class for drawing.
class FullscreenQuad {
  constructor(gl) {
    this.squareVerticesBuffer_ = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.squareVerticesBuffer_);
    gl.bufferData(
        gl.ARRAY_BUFFER, Float32Array.from([-1, -1, 1, -1, -1, 1, 1, 1]),
        gl.STATIC_DRAW);

    this.textureVerticesBuffer_ = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.textureVerticesBuffer_);
    gl.bufferData(
        gl.ARRAY_BUFFER, Float32Array.from([0, 0, 1, 0, 0, 1, 1, 1]),
        gl.STATIC_DRAW);

    this.gl_ = gl;
  }

  // Draws a quad covering the entire bound framebuffer.
  draw() {
    this.gl_.enableVertexAttribArray(0);  // vertex
    this.gl_.bindBuffer(this.gl_.ARRAY_BUFFER, this.squareVerticesBuffer_);
    this.gl_.vertexAttribPointer(0, 2, this.gl_.FLOAT, false, 0, 0);

    this.gl_.enableVertexAttribArray(1);
    this.gl_.bindBuffer(this.gl_.ARRAY_BUFFER, this.textureVerticesBuffer_);
    this.gl_.vertexAttribPointer(1, 2, this.gl_.FLOAT, false, 0, 0);

    this.gl_.drawArrays(this.gl_.TRIANGLE_STRIP, 0, 4);
  }
}

// Utility class for processing a shader.
class GlShaderProcessor {
  constructor(gl, shader) {
    this.quad = new FullscreenQuad(gl);
    this.program = createProgram(gl, PASSTHROUGH_VERTEX_SHADER, shader);
    this.gl = gl;
  }
  startProcessFrame(width, height) {
    const gl = this.gl;
    if (!this.frame || width !== this.frame.width ||
        height !== this.frame.height) {
      this.frame =
          createTextureFrameBuffer(gl, gl.LINEAR, width, height, this.frame);
    }
    gl.viewport(0, 0, width, height);
    gl.scissor(0, 0, width, height);
    this.program.useProgram();
  }
  setUniform(name, vec) {
    const gl = this.gl;
    const loc = this.program.getUniformLocation(name);

    if (typeof vec === 'number') {
      vec = [vec];
    }
    if (vec.length === 1) {
      gl.uniform1fv(loc, vec);
    } else if (vec.length === 2) {
      gl.uniform2fv(loc, vec);
    } else if (vec.length === 3) {
      gl.uniform3fv(loc, vec);
    } else if (vec.length === 4) {
      gl.uniform4fv(loc, vec);
    }
  }
  bindTextures(textures) {
    const gl = this.gl;
    let textureId = 0;
    for (const [name, tex] of textures) {
      const loc = this.program.getUniformLocation(name);
      // Make the textureId unit active.
      gl.activeTexture(gl.TEXTURE0 + textureId);
      // Binds the texture to a TEXTURE_2D target.
      tex.bindTexture();
      // Binds the texture at given location to texture unit textureId.
      gl.uniform1i(loc, textureId);
      textureId++;
    }
  }
  finalizeProcessFrame() {
    this.frame.bindFramebuffer();
    this.quad.draw();
    return this.frame;
  }
}

// A GPU processing step.
class MaskStep {
  constructor(gl) {
    this.proc = new GlShaderProcessor(gl, FRAGMENT_SHADER);
  }
  process(frame, mask) {
    this.proc.startProcessFrame(frame.width, frame.height);

    for (const uniform of shaderUniforms) {
      const name = uniform[0];
      // Prepend k to match shader naming convention for uniforms.
      this.proc.setUniform('k' + name, +STATE[name]);
    }

    this.proc.bindTextures([['frame', frame], ['mask', mask]]);
    return this.proc.finalizeProcessFrame();
  }
}