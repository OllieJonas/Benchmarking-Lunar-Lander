# https://docs.docker.com/samples/library/python/
FROM python:3.9

WORKDIR /app

# basic installs
RUN apt-get update -y \
  && apt-get install -y swig

# copy essential stuff to image
COPY requirements.txt /app

# install project dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
RUN pip3 install gym[Box2D]
RUN pip3 install jupyter

COPY scripts/startup_script.sh /app/scripts/startup_script.sh
COPY rlcw /app/rlcw
COPY config.yml /app/config.yml

# give startup script permission to be ran as an executable
RUN chmod 777 /app/scripts/startup_script.sh

ENV PYTHONPATH=/app
EXPOSE 8888

ENTRYPOINT ["/app/scripts/startup_script.sh"]
