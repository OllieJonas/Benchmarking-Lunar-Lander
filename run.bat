:: boring constants
set IMAGE_NAME=rlcw
set CONTAINER_NAME=rlcw
set CONTAINER_BASE_PATH=""


:: actually run the program
docker build --tag=%IMAGE_NAME% .
docker run --publish 8888:8888 --net=host --name=%CONTAINER_NAME% %IMAGE_NAME%

:: cleanup
docker container stop rlcw
docker container rm rlcw --force

:: copy everything else over
::docker cp %CONTAINER_NAME%:%CONTAINER_BASE_PATH%"/"recordings recordings
::docker cp %CONTAINER_NAME%:%CONTAINER_BASE_PATH%"/"logs logs
::docker cp %CONTAINER_NAME%:%CONTAINER_BASE_PATH%"/"results results

:: leave the terminal open incase something goes wrong
cmd /k

