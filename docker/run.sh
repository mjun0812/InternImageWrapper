#!/bin/bash

cd $(dirname $0)
cd ../
IMAGE_NAME=$(basename $(pwd))
IMAGE_NAME=$(echo $IMAGE_NAME | tr '[:upper:]' '[:lower:]')
USER_ID=`id -u`
GROUP_ID=`id -g`
GROUP_NAME=`id -gn`
USER_NAME=$USER

CMD=${@:-zsh}

if type nvcc > /dev/null 2>&1; then
    # Use GPU
    docker run \
        -it \
        --gpus all \
        --rm \
        --shm-size=128g \
        --hostname $(hostname) \
        --ipc=host \
        --net=host \
        --ulimit memlock=-1 \
        --env DISPLAY=$DISPLAY \
        --env USER_NAME=$USER_NAME \
        --env USER_ID=$USER_ID \
        --env GROUP_NAME=$GROUP_NAME \
        --env GROUP_ID=$GROUP_ID \
        --volume "$(pwd):$(pwd)" \
        --workdir $(pwd) \
        --name "${IMAGE_NAME}-$(date '+%s')" \
        "${IMAGE_NAME}-server:latest" \
        $CMD
else
    # CPU
    docker run \
        -it \
        ${USE_QUEUE} \
        --rm \
        --shm-size=128g \
        --hostname $(hostname) \
        --ipc=host \
        --net=host \
        --ulimit memlock=-1 \
        --env USER_NAME=$USER_NAME \
        --env USER_ID=$USER_ID \
        --env GROUP_NAME=$GROUP_NAME \
        --env GROUP_ID=$GROUP_ID \
        --volume "$(pwd):$(pwd)" \
        --workdir $(pwd) \
        --name "${IMAGE_NAME}-$(date '+%s')" \
        "${IMAGE_NAME}-server:latest" \
        $CMD
fi
