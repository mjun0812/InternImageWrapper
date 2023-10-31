#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
# to lowercase
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')

docker build \
    -t "${IMAGE_NAME}-server:latest" \
    -f docker/Dockerfile .
