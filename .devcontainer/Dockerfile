FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y python3-pip
RUN pip install poetry && poetry config virtualenvs.in-project true
RUN apt-get install -y git
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk
RUN apt install python-is-python3
