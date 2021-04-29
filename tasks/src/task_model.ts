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

import {PackageLoader} from './package_loader';
import {DEFAULT_TFJS_BACKEND, getTFJSModelDependencyPackages, Runtime, Task, TFJSBackend, TFJSModelCommonLoadingOption} from './tasks/common';
import {isWebWorker} from './utils';

/**
 * A model that belongs to one or more machine learning tasks.
 *
 * A task model will be put under its supported task(s) in our main "model
 * index" defined in all_tasks.ts. The index is structured
 * by: {task_name}.{model_name}.{runtime}.
 *
 * For example, for an image classification model Mobilenet from TFJS runtime,
 * its index entry is: ImageClassification.Mobilenet.TFJS.
 *
 * To use this model:
 *
 * // Load.
 * //
 * // This will dynamically load all the packages this model depends on.
 * const model = ImageClassification.Mobilenet.TFJS.load(options);
 *
 * // Inference.
 * //
 * // All models under the same task should have the same API signature. For
 * // example, all the ImageClassification models should have the `classify`
 * // method with the same input and output type.
 * const results = model.classify(img);
 *
 * // Clean up if needed.
 * model.cleanUp();
 *
 * A task model is backed by a "source model". To implement the corresponding
 * task model, a subclass of `TaskModel` needs to be created to:
 *
 * - Provide a set of metadata, such as model names, runtime, etc.
 * - Provide package urls for the source model. `TaskModel` will load the
 *   packages, and return the global namespace variable provided by the source
 *   model.
 * - Implement the `loadSourceModel` method with the loaded global namespace.
 * - Add and implement a task-specific method that uses the loaded source model
 *   to run inference for a given input and return results.
 *
 * To make this process more structured, each task will have a more concrete
 * subclass of `TaskModel` that has the task-specific method defined. All the
 * models under that task will then extend this subclass to make sure they have
 * the same API.
 *
 * For example, for the ImageClassification task, an ImageClassifier class is
 * defined as:
 *
 * class ImageClassifier<N, LO, IO> extends TaskModel<N, LO> {
 *   abstract classify(
 *     img: ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement,
 *     options?: IO): Promise<ImageClassifierResult>;
 * }
 *
 * Then all the models under the ImageClassification task should extend the
 * ImageClassifier class instead.
 *
 * @template N The type of the global namespace provided by the source model.
 * @template LO The type of options used during the model loading process.
 */
export abstract class TaskModel<N, LO> {
  /** The model name. */
  abstract readonly name: string;

  /** The model runtime. */
  abstract readonly runtime: Runtime;

  /**
   * The model version.
   *
   * This is part of the metadata of the model. It doesn't affect which version
   * of the package to load. For that, set the corresponding package url in the
   * `packageUrls` field below.
   *
   * TODO: allow users to dynamically specify which version to load.
   */
  abstract readonly version: string;

  /**
   * Tasks that the model supports.
   *
   * This needs to match the location(s) of this model in the main index in
   * all_tasks.ts file. A test will run to make sure of this.
   */
  abstract readonly supportedTasks: Task[];

  /**
   * URLs of packages to load for the model.
   *
   * Each element in the array is an array of package urls (package set).
   * Package sets are loaded one after another. Within each set, packages are
   * loaded at the same time.
   *
   * For example, if packageUrls = [ [url1, url2], [url3] ], then url1 and
   * url2 are loaded together first. url3 will then be loaded when url1 and
   * url2 are both done loading.
   *
   * Note: for TFJS model, only the model package itself needs to be set
   * here. The TFJS related dependencies (e.g. core, backends, etc) will be
   * added automatically.
   */
  abstract readonly packageUrls: Array<string[]>;

  /**
   * The global namespace of the source model.
   *
   * For example, for TFJS mobilenet model, this would be 'mobilenet'.
   */
  abstract readonly sourceModelGlobalNs: string;

  /**
   * Callback to wait for (await) after all packages are loaded.
   *
   * This is necessary for certain models where certain conditions need to be
   * met before using the model.
   */
  readonly postPackageLoadingCallback?: () => Promise<void> = undefined;

  /** A loader for loading packages. */
  private readonly packageLoader = new PackageLoader<N>();

  /**
   * Loads the task model with the given options.
   *
   * @param options Options to use when loading the model.
   * @returns The loaded instance.
   */
  async load(options?: LO): Promise<this> {
    const sourceModelGlobal = await this.loadSourceModelGlobalNs(options);
    // Wait for the callback if set.
    if (this.postPackageLoadingCallback) {
      await this.postPackageLoadingCallback();
    }
    // For tfjs models, we automatically wait for "tf.ready()" before
    // proceeding. Subclasses don't need to set postPackageLoadingCallback
    // explicitly.
    if (this.runtime === Runtime.TFJS) {
      // tslint:disable-next-line:no-any
      const global: any = isWebWorker() ? self : window;
      const tfjsOptions = options as {} as TFJSModelCommonLoadingOption;
      const backend: TFJSBackend =
          tfjsOptions ? tfjsOptions.backend : DEFAULT_TFJS_BACKEND;
      await (global['tf'].setBackend(backend) as Promise<void>);
    }
    await this.loadSourceModel(sourceModelGlobal, options);
    return this;
  }

  /**
   * Cleans up resources if necessary.
   *
   * It does nothing by default. A subclass should override this method if it
   * has any clean up related tasks to do.
   */
  cleanUp() {}

  /**
   * Loads the global namespace of the source model by loading its packages
   * specified in the `packageUrls` field above.
   *
   * It is typically not necessary for subclasses to override this method as
   * long as they set the `packageUrls` and `sourceModelGlobalNs` field above.
   */
  protected async loadSourceModelGlobalNs(options?: LO): Promise<N> {
    const packages: Array<string[]> = [];

    // Add TFJS dependencies for TFJS models.
    if (this.runtime === Runtime.TFJS) {
      const tfjsOptions = options as {} as TFJSModelCommonLoadingOption;
      packages.push(...getTFJSModelDependencyPackages(
          tfjsOptions ? tfjsOptions.backend : undefined));
    }

    // Load packages.
    packages.push(...this.packageUrls);
    return await this.packageLoader.loadPackagesAndGetGlobalNs(
        this.sourceModelGlobalNs, packages);
  }

  /**
   * Loads the source model with the given global namespace and options.
   *
   * A subclass should implement this method and use the
   * `sourceModelGlobalNs` to load the source model. The loaded source model
   * should be stored in a property in the subclass that can then be used during
   * the inference.
   *
   * Note that the subclass should *NOT* use the namespace from the import
   * statement to implement this method (e.g. the "mobilenet" variable in
   * `import * as mobilenet from '@tensorflow-models/mobilenet`). Instead, only
   * `sourceModelGlobalNs` should be used, which should have the same type as
   * the imported namespace. The imported namespace can only be used to
   * reference types. This makes sure that the code from the source model is NOT
   * bundled in the final binary.
   */
  protected async loadSourceModel(sourceModelGlobalNs: N, options?: LO) {
    throw new Error('not implemented');
  }
}
