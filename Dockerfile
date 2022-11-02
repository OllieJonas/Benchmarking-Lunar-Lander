# https://docs.docker.com/samples/library/python/
FROM python:3.9

WORKDIR /app
RUN apt-get update

RUN pip3 install --upgrade pip
RUN pip3 install gym[All]

ENTRYPOINT []