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

import {isWebWorker} from './utils';

/** Function type declaration for loading script in a webworker. */
declare function importScripts(...urls: string[]): void;

/** A package loader. */
export class PackageLoader<G> {
  private static readonly promises: {[url: string]: Promise<void>} = {};

  /**
   * Loads the given packages and returns the loaded global namespace.
   *
   * Each element in `packageUrls` is an array of package urls (package set).
   * Package sets are loaded one after another. Within each set, packages are
   * loaded at the same time.
   *
   * For example, if packageUrls = [ [url1, url2], [url3] ], then url1 and
   * url2 are loaded together first. url3 will then be loaded when url1 and
   * url2 are both done loading.
   */
  async loadPackagesAndGetGlobalNs(
      namespace: string, packageUrls: Array<string[]>,
      curIndex = 0): Promise<G> {
    if (curIndex >= packageUrls.length) {
      // tslint:disable-next-line:no-any
      const global: any = isWebWorker() ? self : window;
      return global[namespace] as G;
    }
    await Promise.all(
        packageUrls[curIndex].map(name => this.loadPackage(name)));
    return await this.loadPackagesAndGetGlobalNs(
        namespace, packageUrls, curIndex + 1);
  }

  private async loadPackage(packageUrl: string): Promise<void> {
    // Don't load a package if the package is being loaded or has already been
    // loaded.
    if (PackageLoader.promises[packageUrl]) {
      return PackageLoader.promises[packageUrl];
    }

    // From webworker.
    if (typeof window === 'undefined') {
      importScripts(packageUrl);
      PackageLoader.promises[packageUrl] = Promise.resolve();
    }
    // From webpage.
    else {
      const script = document.createElement('script');
      const promise = new Promise<void>((resolve, reject) => {
        script.onerror = () => {
          reject();
          document.head.removeChild(script);
        };
        script.onload = () => {
          resolve();
        };
      });
      document.head.appendChild(script);
      script.setAttribute('src', packageUrl);
      PackageLoader.promises[packageUrl] = promise;
    }
    return PackageLoader.promises[packageUrl];
  }
}
