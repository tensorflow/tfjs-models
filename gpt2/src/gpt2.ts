/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export class GPT2 {
  // @pforderique feel free to change the API
  // This api should be friendly to JS users.
  async generate(input: string): Promise<string> {
    // Fake delay for where the model will run.
    await new Promise((resolve) => {
      setTimeout(resolve, 300);
    });

    console.log(`got input '${input}'`);
    return ' the park';
  }
}
