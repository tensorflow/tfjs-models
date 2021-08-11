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

import {ChangeDetectionStrategy, Component} from '@angular/core';
import {Router} from '@angular/router';
import {select, Store} from '@ngrx/store';
import {UrlParamKey} from 'src/app/common/types';
import {ModelItem} from 'src/app/models/model_item';
import {selectAllModelItems, selectSelectedModelItemId} from 'src/app/store/selectors';
import {AppState} from 'src/app/store/state';

/**
 * The tree to show all models grouped by tasks.
 */
@Component({
  selector: 'navigation-tree',
  templateUrl: './navigation_tree.component.html',
  styleUrls: ['./navigation_tree.component.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class NavigationTree {
  items$ = this.store.pipe(select(selectAllModelItems));
  selectedItemId$ = this.store.pipe(select(selectSelectedModelItemId));

  constructor(
      private store: Store<AppState>,
      private readonly router: Router,
  ) {}

  handleClick(item: ModelItem) {
    this.router.navigate([], {
      queryParams: {
        [UrlParamKey.MODEL_ITEM_ID]: item.id,
      },
      queryParamsHandling: 'merge',
    });
  }
}
