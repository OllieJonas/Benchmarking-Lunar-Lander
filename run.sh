!#/bin/bash

# boring constants
IMAGE_NAME=rlcw
CONTAINER_NAME=rlcw
CONTAINER_BASE_PATH=""

# actually run the program
docker build --tag=$IMAGE_NAME .
docker run --name=$CONTAINER_NAME $IMAGE_NAME

## copy everything else over
#docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"recordings recordings
#docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"logs logs
#docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"results results


