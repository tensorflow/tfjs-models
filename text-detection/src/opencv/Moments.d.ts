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

export class Moments {
  readonly m00: number;
  readonly m10: number;
  readonly m01: number;
  readonly m20: number;
  readonly m11: number;
  readonly m02: number;
  readonly m30: number;
  readonly m21: number;
  readonly m12: number;
  readonly m03: number;
  readonly mu20: number;
  readonly mu11: number;
  readonly mu02: number;
  readonly mu30: number;
  readonly mu21: number;
  readonly mu12: number;
  readonly mu03: number;
  readonly nu20: number;
  readonly nu11: number;
  readonly nu02: number;
  readonly nu30: number;
  readonly nu21: number;
  readonly nu12: number;
  readonly nu03: number;
  huMoments(): number[];
}
