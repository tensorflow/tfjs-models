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
import {ensureTFJSBackend, getTFJSModelDependencyPackages, Runtime, Task, TFJSModelCommonLoadingOption} from './tasks/common';

/**
 * A model that belongs to one or more machine learning tasks.
 *
 * In TFJS Task API, models are grouped into different tasks (e.g.
 * ImageClassification, PoseDetection, etc). Each model is a "task model", and
 * it is implemented as an instance of the `TaskModel` interface defined here.
 *
 * A task model is loaded through the corresponding `TaskModelLoader` (see below
 * for more details). In order to help users easily locate various loaders, we
 * organize them into a central "model loader index" defined in all_tasks.ts.
 * The index structure is {task_name}.{model_name}.{runtime}. For example:
 * `ImageClassification.MobileNet.TFJS`.
 *
 *
 * Basic usage:
 *
 * // Load the model.
 * //
 * // - This will dynamically load all the packages that the model depends on.
 * // - Different models can have different loading options.
 * const model = await ImageClassification.Mobilenet.TFJS.load(
 *   {backend: 'wasm'});
 *
 * // Inference.
 * //
 * // - All the models under the same task should have the same API signature.
 * //   For example, all the ImageClassification models should have the
 * //   `classify` method with the same input and output type.
 * // - Different models can have different inference options.
 * const results = await model.classify(img, {topK: 5});
 *
 * // Clean up if needed.
 * model.cleanUp();
 */
export interface TaskModel {
  /**
   * Cleans up resources if necessary.
   */
  cleanUp(): void;
}

/** Metadata for a task model. */
export interface TaskModelMetadata {
  /** Human-readable model name. */
  name: string;

  /** The model description. Can have simple HTML tags. */
  description?: string;

  /** Resource urls (e.g. github page, docs, etc) indexed by names. */
  resourceUrls?: {[name: string]: string};

  /** The model runtime. */
  runtime: Runtime;

  /**
   * The model version.
   *
   * This should be used to construct the main package url in the `packageUrls`
   * field below so that the package version matches the version specified here.
   *
   * TODO: allow users to dynamically specify which version to load.
   */
  version: string;

  /**
   * Tasks that the model supports.
   *
   * This needs to match the location(s) of this model loader in the main index
   * in all_tasks.ts file. A test will run to make sure of this.
   */
  supportedTasks: Task[];
}

/**
 * A loader that loads a task model.
 *
 * A task model is backed by a "source model" that does the actual work, so the
 * main task of a model loader is to load this source model and "transform" it
 * into the target task model. To do that, the client that implements the
 * `TaskModelLoader` needs to:
 *
 * - Provide a set of metadata, such as model name, runtime, etc.
 *
 * - Provide package urls and global namespace name for the source model.
 *   `TaskModelLoader` will load the packages, and return the global namespace
 *   variable provided by the source model.
 *
 * - Implement the `transformSourceModel` method to load the source model using
 *   the global namespace variable from the previous step, and transform the
 *   source model to the corresponding task model.
 *
 *   To make this process more structured, each task will have a more concrete
 *   interface of `TaskModel` with the task-specific method defined. All the
 *   models under the task will transform to this interface to make sure they
 *   have the same API signature. See tasks/image_classification/common.ts for
 *   an example.
 *
 * @template N The type of the global namespace provided by the source model.
 * @template LO The type of options used during the model loading process.
 * @template M The type of the target task model to transform to.
 */
export abstract class TaskModelLoader<N, LO, M> {
  /** Metadata */
  abstract readonly metadata: TaskModelMetadata;

  /**
   * URLs of packages to load for the model.
   *
   * `packageUrls` allows you to express package dependencies. Each element in
   * it is an array of package urls (package set). Package sets are downloaded
   * and evaluated one after another. Within each set, packages are downloaded
   * in parallel, and evaluated in the order of download finish time. Packages
   * in a package set should yeild the same outcome no matter what order they
   * are evaluated in.
   *
   * For example, if packageUrls = [ [url1, url2], [url3] ], then url1 and
   * url2 will be downloaded in parallel first. They will then be
   * evaluated by the browser in sequence depends on which one is downloaded
   * first. After that is done, the package loader will start downloading
   * and evaluating url3.
   *
   * Note: for TFJS model, only the model package itself needs to be set
   * here. The TFJS related dependencies (e.g. core, backends, etc) will be
   * added automatically.
   */
  abstract readonly packageUrls: string[][];

  /**
   * The global namespace of the source model.
   *
   * For example, for TFJS mobilenet model, this would be 'mobilenet'.
   */
  abstract readonly sourceModelGlobalNamespace: string;

  /**
   * Callback to wait for (await) after all packages are loaded.
   *
   * This is necessary for models where certain conditions need to be met before
   * using the model.
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
  async load(options?: LO): Promise<M> {
    // Load the packages to get the global namespace ready.
    const sourceModelGlobal =
        await this.loadSourceModelGlobalNamespace(options);

    // Wait for the callback to resolve if set.
    if (this.postPackageLoadingCallback) {
      await this.postPackageLoadingCallback();
    }
    // For tfjs models, we automatically wait for the backend to be set before
    // proceeding. Subclasses don't need to do worry about this.
    if (this.metadata.runtime === Runtime.TFJS) {
      await ensureTFJSBackend(options as {} as TFJSModelCommonLoadingOption);
    }

    // Load the source model using the global namespace variable loaded above.
    return this.transformSourceModel(sourceModelGlobal, options);
  }

  /**
   * Loads the global namespace of the source model by loading its packages
   * specified in the `packageUrls` field above.
   *
   * It is typically not necessary for subclasses to override this method as
   * long as they set the `packageUrls` and `sourceModelGlobalNamespace` field
   * above.
   */
  protected async loadSourceModelGlobalNamespace(options?: LO): Promise<N> {
    const packages: string[][] = [];

    // Add TFJS dependencies for TFJS models.
    if (this.metadata.runtime === Runtime.TFJS) {
      const tfjsOptions = options as {} as TFJSModelCommonLoadingOption;
      packages.push(...getTFJSModelDependencyPackages(
          tfjsOptions ? tfjsOptions.backend : undefined));
    }

    // Load packages.
    packages.push(...this.packageUrls);
    return this.packageLoader.loadPackagesAndGetGlobalNamespace(
        this.sourceModelGlobalNamespace, packages);
  }

  /**
   * Loads the source model and transforms it to the corresponding task model.
   *
   * A subclass should implement this method and use the
   * `sourceModelGlobalNamespace` to load the source model. Note that the
   * subclass should *NOT* use the namespace from the import statement to
   * implement this method (e.g. the "mobilenet" variable in `import * as
   * mobilenet from
   * '@tensorflow-models/mobilenet`). Instead, only `sourceModelGlobalNamespace`
   * should be used, which should have the same type as the imported namespace.
   * The imported namespace can only be used to reference types. This makes sure
   * that the code from the source model is NOT bundled in the final binary.
   */
  protected async transformSourceModel(
      sourceModelGlobalNamespace: N, options?: LO): Promise<M> {
    throw new Error('not implemented');
  }
}
