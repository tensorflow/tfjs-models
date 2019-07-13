// MIT License
// Copyright (c) 2017 Vincent Mühler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import {Vec3} from './Vec3.d';

export class Vec {
  absdiff(otherVec: Vec): Vec;
  add(otherVec: Vec): Vec;
  at(index: number): number;
  cross(): Vec3;
  div(s: number): Vec;
  exp(): Vec;
  hDiv(otherVec: Vec): Vec;
  hMul(otherVec: Vec): Vec;
  mean(): Vec;
  mul(s: number): Vec;
  norm(): number;
  sqrt(): Vec;
  sub(otherVec: Vec): Vec;
}
