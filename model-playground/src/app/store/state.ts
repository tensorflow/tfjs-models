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

import {ActionReducerMap} from '@ngrx/store';

import {ModelItem} from '../models/model_item';

import {allModelItemsReducer} from './reducers';

/** The main app state. */
export interface AppState {
  /** All registered ModelItem objects. */
  allModelItems: ModelItem[];
}

/** The initial app state. */
export const initialState: AppState = {
  allModelItems: [],
};

/** Reducers for each app state field. */
export const appReducers: ActionReducerMap<AppState> = {
  allModelItems: allModelItemsReducer,
};
