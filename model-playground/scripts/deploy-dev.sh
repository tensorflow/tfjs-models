#!/bin/bash

# Build and deploy the app to the `model-playground-demo` Google Storage bucket.

set -e

# Use date+time as an unique identifier for the current demo.
datetime=$(date '+%Y-%m-%d-%H-%M-%S')

# Build and copy angular app to the corresponding Google Storage location.
base_url="/model-playground-demo/${datetime}/"
yarn build --base-href ${base_url} --deploy-url ${base_url}
gsutil -m cp dist/* gs://model-playground-demo/${datetime}/

# Output the url to access to demo.
echo "-------------------------------------------------------------------------"
echo "Demo deployed: https://storage.googleapis.com/model-playground-demo/${datetime}/index.html"
