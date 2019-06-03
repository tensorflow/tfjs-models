/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import 'bulma/css/bulma.css';
import { SemanticSegmentation } from '@tensorflow-models/deeplab';

const deeplab = {
    pascal: undefined,
    cityscapes: undefined,
    ade20k: undefined,
};

const initialiseModels = () => {
    Object.keys(deeplab).forEach(modelName => {
        const model = new SemanticSegmentation(modelName);
        deeplab[modelName] = model;
    });
};

const processImage = file => {
    if (!file.type.match('image.*')) {
        return;
    }
    const reader = new FileReader();
    reader.onload = event => {
        const image = document.createElement('img');
        image.src = event.target.result;
        document.appendChild(image);
    };
};

const processImages = event => {
    const files = event.target.files;
    Array.from(files).forEach(processImage);
};

const getImage = () => {
    const image = document.getElementById('input-image');
    if (!image) {
        alert('Please pass a valid input image.');
    }
    return image;
};

const displayResult = result => {};

const runModel = async modelName => {
    const model = deeplab[modelName];
    const input = getImage();
    if (input.complete && input.naturalHeight !== 0) {
        displayResult(await model.predict(input));
    } else {
        input.onload = async () => {
            displayResult(await model.predict(input));
        };
    }
};

window.onload = initialiseModels;

const uploader = document.getElementById('upload-image');
uploader.addEventListener('change', processImages);
