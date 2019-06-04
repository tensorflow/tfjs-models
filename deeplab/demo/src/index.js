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
import pascalExampleImage from './examples/pascal.jpg';
import cityscapesExampleImage from './examples/cityscapes.jpg';
import ade20kExampleImage from './examples/ade20k.jpg';

const deeplab = {
    pascal: undefined,
    cityscapes: undefined,
    ade20k: undefined,
};

const deeplabExampleImages = {
    pascal: pascalExampleImage,
    cityscapes: cityscapesExampleImage,
    ade20k: ade20kExampleImage,
};

const initialiseModels = () => {
    Object.keys(deeplab).forEach(modelName => {
        const model = new SemanticSegmentation(modelName);
        deeplab[modelName] = model;
        const toggler = document.getElementById(`toggle-${modelName}-image`);
        toggler.onclick = () => setImage(deeplabExampleImages[modelName]);
        const runner = document.getElementById(`run-${modelName}`);
        runner.onclick = () => runDeeplab(modelName);
    });
};

const setImage = src => {
    const image = document.getElementById('input-image');
    image.src = src;
    const imageContainer = document.getElementById('input-card');
    imageContainer.classList.remove('is-invisible');
};

const processImage = file => {
    if (!file.type.match('image.*')) {
        return;
    }
    const reader = new FileReader();
    reader.onload = event => {
        setImage(event.target.result);
    };
    reader.readAsDataURL(file);
};

const processImages = event => {
    const files = event.target.files;
    Array.from(files).forEach(processImage);
};

const displaySegmentationMap = ([height, width, segmentationMap]) => {
    console.log(segmentationMap);
    const canvas = document.getElementById('output-image');
    const ctx = canvas.getContext('2d');
    const segmentationMapData = new ImageData(segmentationMap, width, height);
    ctx.putImageData(segmentationMapData, 0, 0);
    const imageContainer = document.getElementById('output-card');
    imageContainer.classList.remove('is-invisible');
};

const runDeeplab = async modelName => {
    const input = document.getElementById('input-image');
    if (!input.src || !input.src.length || input.src.length === 0) {
        alert('Please load an image first.');
        return;
    }

    const model = deeplab[modelName];
    if (input.complete && input.naturalHeight !== 0) {
        displaySegmentationMap(await model.predict(input));
    } else {
        input.onload = async () => {
            displaySegmentationMap(await model.predict(input));
        };
    }
    console.log('DONE');
};

window.onload = initialiseModels;

const uploader = document.getElementById('upload-image');
uploader.addEventListener('change', processImages);
