# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import division
from __future__ import print_function

import os
import shutil
import struct
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

import prep_wavs


class PrepWavsTest(unittest.TestCase):

  def setUp(self):
    self._tmpDir = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self._tmpDir):
      shutil.rmtree(self._tmpDir)

  def testReadAsFloats(self):
    wav_path = os.path.join(self._tmpDir, 'test.wav')
    wavfile.write(wav_path, 44100, np.zeros([100], dtype=np.int16))
    fs, signal = prep_wavs.read_as_floats(wav_path)
    self.assertEqual(44100, fs)
    self.assertEqual(100, len(signal))
    self.assertEqual(np.float32, signal.dtype)
    self.assertEqual(0.0, signal[0])
    self.assertEqual(0.0, signal[-1])

  def testReadAndResampleAsFloats(self):
    wav_path = os.path.join(self._tmpDir, 'test.wav')
    wavfile.write(wav_path, 22050, 100 * np.ones([100], dtype=np.int16))
    signal = prep_wavs.read_and_resample_as_floats(wav_path, 44100)
    self.assertEqual(100 * 2, len(signal))
    self.assertEqual(np.float32, signal.dtype)
    self.assertAlmostEqual(100 / 32768, signal[0])
    self.assertAlmostEqual(100 / 32768, signal[-1])

  def testLoadAndNormalizeWaveform(self):
    wav_path = os.path.join(self._tmpDir, 'test.wav')
    wavfile.write(wav_path, 22050, 100 * np.ones([100], dtype=np.int16))
    signal = prep_wavs.load_and_normalize_waveform(wav_path, 44100, 32)
    self.assertEqual(192, len(signal))
    self.assertEqual(np.float32, signal.dtype)
    self.assertAlmostEqual(100 / 32768, signal[0])
    self.assertAlmostEqual(100 / 32768, signal[-1])

  def testConvert(self):
    in_wav_path = os.path.join(self._tmpDir, 'test_in.wav')
    wavfile.write(in_wav_path, 22050, 100 * np.ones([100], dtype=np.int16))

    out_dat_path = os.path.join(self._tmpDir, 'test_out.dat')
    length = prep_wavs.convert(in_wav_path, 44100, 32, out_dat_path)
    self.assertEqual(192, length)
    self.assertTrue(os.path.isfile(out_dat_path))
    self.assertEqual(192 * 4, os.stat(out_dat_path).st_size)
    with open(out_dat_path, 'rb') as f:
      data = struct.unpack('=192f', f.read())
      self.assertAlmostEqual(100 / 32768, data[0])
      self.assertAlmostEqual(100 / 32768, data[-1])


if __name__ == '__main__':
  unittest.main()
