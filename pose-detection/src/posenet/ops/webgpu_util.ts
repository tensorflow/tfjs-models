/**
 * @license
 * Copyright 2023 Google LLC.
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

export function getMainHeaderString(): string;
export function getMainHeaderString(index: string): string;
export function getMainHeaderString(...params: string[]): string {
  let snippet: string;
  switch (params.length) {
    case 0:
      snippet = `fn main() `;
      break;
    case 1:
      snippet = `fn main(${params[0]} : i32)`;
      break;
    default:
      throw Error('Unreachable');
  }
  return snippet;
}
