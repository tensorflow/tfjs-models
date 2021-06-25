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

/**
 * A loader for dynamically loading a set of packages (with dependencies) and
 * returning the global namespace variable initialized by those packages.
 *
 * @template G The type of the global namespace variable.
 */
export class PackageLoader<G> {
  private static readonly promises: {[url: string]: Promise<void>} = {};

  /**
   * Loads the given packages and returns the loaded global namespace.
   *
   * See `TaskModelLoader.packageUrls` for more info about how to use it to
   * specify packages with dependencies.
   *
   * TODO: use a design similar to tfjs' `Platform` to provide different
   * implmentations for dynamic package loading, global, etc.
   */
  async loadPackagesAndGetGlobalNamespace(
      namespace: string, packageUrls: string[][], curIndex = 0): Promise<G> {
    if (curIndex >= packageUrls.length) {
      // tslint:disable-next-line:no-any
      const global: any = isWebWorker() ? self : window;
      const globalNamespace = global[namespace] as G;
      if (!globalNamespace) {
        throw new Error(`Global namespace '${
            namespace}' not set after loading packages ${packageUrls}`);
      }
      return globalNamespace;
    }
    await Promise.all(
        packageUrls[curIndex].map(name => this.loadPackage(name)));
    return this.loadPackagesAndGetGlobalNamespace(
        namespace, packageUrls, curIndex + 1);
  }

  private async loadPackage(packageUrl: string): Promise<void> {
    // Don't load a package if the package is being loaded or has already been
    // loaded.
    if (PackageLoader.promises[packageUrl]) {
      return PackageLoader.promises[packageUrl];
    }

    // From webworker.
    if (isWebWorker()) {
      importScripts(packageUrl);
      PackageLoader.promises[packageUrl] = Promise.resolve();
    }
    // From webpage.
    else {
      const script = document.createElement('script');
      script.crossOrigin = 'anonymous';
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
