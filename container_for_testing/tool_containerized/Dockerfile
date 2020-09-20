FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install some dependencies
RUN apt-get update && \
    apt-get install -y \
      python3.8 \
      python3-pip \
      && \
    apt-get install -y ffmpeg && \
    apt-get install -y pkg-config && \
    mkdir data &&\
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# python dependencies
COPY docker_project/requirements.txt .

RUN pip3 install -r requirements.txt

COPY docker_project/ docker_project/

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR .
# octave and octave pkgs
RUN apt-get update && \
    apt-get install -y octave &&\
    apt-get install -y octave-control octave-image octave-io octave-optim octave-signal octave-statistics &&\
    apt-get install -y libavutil-dev libavformat-dev libswscale-dev libavcodec-dev liboctave-dev &&\
    octave ./docker_project/octave_pkgs/setup.m &&\
    apt-get clean && \
    apt-get autoremove

