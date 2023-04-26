# Select base image with cuda
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu18.04

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install eseential packages
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    curl \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install wget.
RUN apt-get update && \
    apt-get install -y wget

## INPUT
# Set app name:
ARG APP_NAME=structure
ARG MODEL_NAME=foldingdiff

################################################
# Boiler plate setup
ARG APP_PATH=/$APP_NAME
ARG POETRY_VERSION=1.2.0a2
ARG NEW_RELIC_ACCOUNT_ID=3437176
ARG NEW_RELIC_API_KEY

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH \
ENV PATH /opt/conda/bin:$PATH

# Copy all the files from the repo to the container
COPY . ${APP_PATH}

# Make the working directory the app path
WORKDIR ${APP_PATH}

# Install the conda environment
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "foldingdiff", "/bin/bash", "-c"]
RUN pip install -e .

# Download the data using data/download_cath.sh, but run it locally
WORKDIR ${APP_PATH}/data
RUN bash download_cath.sh

# install FASPR package
RUN pip install pybind11
WORKDIR ${APP_PATH}/faspr
RUN pip install -e .

# Reset the working directory
WORKDIR ${APP_PATH}