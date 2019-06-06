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

const initialiseModels = async () => {
    Object.keys(deeplab).forEach(modelName => {
        const model = new SemanticSegmentation(modelName);
        deeplab[modelName] = model;
        const toggler = document.getElementById(`toggle-${modelName}-image`);
        toggler.onclick = () => setImage(deeplabExampleImages[modelName]);
        const runner = document.getElementById(`run-${modelName}`);
        runner.onclick = async () => {
            runner.classList.add('is-loading');
            await sleep(100);
            runDeeplab(modelName);
            runner.classList.remove('is-loading');
        };
    });
    const uploader = document.getElementById('upload-image');
    uploader.addEventListener('change', processImages);
    status('Initialised models, waiting for input...');
};

const setImage = src => {
    const outputContainer = document.getElementById('output-card');
    outputContainer.classList.add('is-invisible');
    const legendContainer = document.getElementById('legend-card');
    legendContainer.classList.add('is-invisible');
    const image = document.getElementById('input-image');
    image.src = src;
    const imageContainer = document.getElementById('input-card');
    imageContainer.classList.remove('is-invisible');
    status('Waiting until the model is picked...');
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

const displaySegmentationMap = deeplabOutput => {
    const [legend, height, width, segmentationMap] = deeplabOutput;
    const canvas = document.getElementById('output-image');
    const ctx = canvas.getContext('2d');

    const outputContainer = document.getElementById('output-card');
    outputContainer.classList.remove('is-invisible');
    const segmentationMapData = new ImageData(segmentationMap, width, height);
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.width = width;
    canvas.height = height;
    ctx.putImageData(segmentationMapData, 0, 0);

    const legendList = document.getElementById('legend');
    while (legendList.firstChild) {
        legendList.removeChild(legendList.firstChild);
    }

    // Work around the parcel failure with Object.keys
    for (const label in legend) {
        if (legend.hasOwnProperty(label)) {
            const tag = document.createElement('span');
            tag.innerHTML = label;
            const [red, green, blue] = legend[label];
            tag.classList.add('column');
            tag.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
            tag.style.padding = '1em';
            tag.style.margin = '1em';
            tag.style.color = '#ffffff';

            legendList.appendChild(tag);
        }
    }

    const legendContainer = document.getElementById('legend-card');
    legendContainer.classList.remove('is-invisible');
};

const sleep = ms => {
    return new Promise(resolve => setTimeout(resolve, ms));
};

const status = message => {
    const statusMessage = document.getElementById('status-message');
    statusMessage.innerText = message;
};

const runDeeplab = modelName => {
    const initialisationStart = performance.now();
    const input = document.getElementById('input-image');
    if (!input.src || !input.src.length || input.src.length === 0) {
        alert('Please load an image first.');
        return;
    }

    const model = deeplab[modelName];
    if (input.complete && input.naturalHeight !== 0) {
        model.predict(input).then(output => {
            displaySegmentationMap(output);
            status(`Ran in ${performance.now() - initialisationStart} ms`);
        });
    } else {
        input.onload = () => {
            model.predict(input).then(output => {
                displaySegmentationMap(output);
                status(`Ran in ${performance.now() - initialisationStart} ms`);
            });
        };
    }
};

window.onload = initialiseModels;
