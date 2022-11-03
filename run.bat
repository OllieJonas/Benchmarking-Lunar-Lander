:: boring constants
set IMAGE_NAME=rlcw
set CONTAINER_NAME=rlcw
set CONTAINER_BASE_PATH=""

:: even more boring dir setup
mkdir recordings
mkdir logs
mkdir results
mkdir policies


:: mount a volume to copy logs, recordings and results to out of the container
docker
:: actually run the program
docker build --tag=%IMAGE_NAME% .
docker run --name=%CONTAINER_NAME% %IMAGE_NAME%

cmd /k
:: copy everything else over
::docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"recordings recordings
::docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"logs logs
::docker cp $CONTAINER_NAME:$CONTAINER_BASE_PATH"/"results results


