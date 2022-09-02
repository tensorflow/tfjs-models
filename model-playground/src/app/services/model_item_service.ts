/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';

import {MobileNetTfjs} from '../models/image_classification/mobilenet_tfjs';
import {MobileNetTfLite} from '../models/image_classification/mobilenet_tflite';
import {CocoSsdTfjs} from '../models/object_detection/cocossd_tfjs';
import {addModelItemsFromInit} from '../store/actions';
import {AppState} from '../store/state';

/**
 * Service for model item related tasks.
 */
@Injectable({
  providedIn: 'root',
})
export class ModelItemService {
  constructor(
      private readonly store: Store<AppState>,
  ) {}

  /** Registers all model items. */
  registerAllModelItems() {
    this.store.dispatch(addModelItemsFromInit({
      items: [
        new MobileNetTfjs(),
        new MobileNetTfLite(),
        new CocoSsdTfjs(),
      ]
    }));
  }
}
