FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y python3-pip
RUN pip install poetry && poetry config virtualenvs.in-project true
RUN apt-get install -y git
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk
RUN apt install python-is-python3

###############################################################################

# FROM gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010
# LABEL maintainer "Matt Johnson <mattjj@google.com>"

# WORKDIR /
# # TODO(skyewm): delete the following line when no longer necessary.
# RUN rm -f /etc/apt/sources.list.d/jonathonf-ubuntu-python-3_6-xenial.list
# RUN apt-get update
# RUN apt-get install libffi-dev


# # Change the CUDA version if it doesn't match the installed version.
# ARG JAX_CUDA_VERSION=11.1
# COPY install_cuda.sh /install_cuda.sh
# RUN chmod +x /install_cuda.sh
# RUN /bin/bash -c 'if [[ ! "$CUDA_VERSION" =~ ^$JAX_CUDA_VERSION.*$ ]]; then \
#   /install_cuda.sh $JAX_CUDA_VERSION; \
#   fi'

# # RUN apt install -y python3-pip python-is-python3
# RUN pip install poetry && poetry config virtualenvs.in-project true
# RUN apt-get install -y git
# ARG DEBIAN_FRONTEND=noninteractive
# ENV TZ=Europe/London
# RUN apt-get install -y python3-tk

