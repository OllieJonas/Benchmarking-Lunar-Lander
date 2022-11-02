# https://docs.docker.com/samples/library/python/
FROM python:3.9

WORKDIR /app

RUN apt-get update
RUN apt-get install -y swig
RUN apt-get install -y cmake

COPY requirements.txt /app

RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install gym[All]

COPY scripts/startup_script.sh /app/scripts/startup_script.sh
COPY rlcw /app/rlcw

RUN chmod 777 /app/scripts/startup_script.sh

ENTRYPOINT ["/app/scripts/startup_script.sh"]