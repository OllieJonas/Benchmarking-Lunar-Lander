!#/bin/bash

docker build --tag=rlcw .
docker run -it --name=rlcw rlcw python