/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export const config = {
  // #TODO(tfjs) : Replace this URL after you host the model
  BASE_PATH: 'https://storage.googleapis.com/gsoc-tfjs/models/text-detection',
  // #TODO(tfjs) : Remove this after the model version is finalized
  MODEL_VERSION: 'psenet-rc185-v1',
  RESIZE_LENGTH: 256,
  MIN_PIXEL_SALIENCE: 1,
  MIN_TEXT_CONFIDENCE: 0.94,
  MIN_TEXTBOX_AREA: 10,
  DEBUG: false
};
