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

import {getSelectors, RouterReducerState} from '@ngrx/router-store';
import {createFeatureSelector, createSelector} from '@ngrx/store';

import {UrlParamKey} from '../common/types';

import {AppState} from './state';

const selectRouter = createFeatureSelector<RouterReducerState>('router');
const selectQueryParams = getSelectors(selectRouter).selectQueryParams;

/** Selector to select all model items. */
export const selectAllModelItems = (state: AppState) => {
  return state.allModelItems;
};

/** Selector to select the id of the currently selected model item from URL. */
export const selectSelectedModelItemId =
    createSelector(selectQueryParams, (params) => {
      if (!params) {
        return '';
      }
      return params[UrlParamKey.MODEL_ITEM_ID];
    });
