FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system packages and dependencies for Python build
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        curl wget gnupg2 \
        git tmux htop sudo \
        build-essential \
        pciutils \
        pkg-config libfreetype6-dev libpng-dev \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libncurses5-dev libncursesw5-dev xz-utils \
        tk-dev libffi-dev liblzma-dev make && \
    apt-get clean

# Install Python 3.7.17 from source
WORKDIR /usr/src
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev \
    libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev \
    libexpat1-dev liblzma-dev tk-dev uuid-dev && \
    wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz && \
    tar xzf Python-3.7.17.tgz && \
    cd Python-3.7.17 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    rm -rf /usr/src/Python-3.7.17*

# Make Python 3.7 the default for this environment (optional)
RUN ln -sfn /usr/local/bin/python3.7 /usr/bin/python3 && \
    ln -sfn /usr/local/bin/pip3.7 /usr/bin/pip3

# Upgrade pip to a specific version for reproducibility
RUN pip3 install --upgrade pip==20.2

# Work directory
WORKDIR /tmp/

# Copy and patch requirements
COPY ./requirements.txt .
RUN echo '\ntensorflow==1.13.1' >> requirements.txt
RUN pip3 install -r requirements.txt

# Create a non-root user
ARG USERNAME=ruggeri
ARG USER_UID=2565
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

ARG DOCKER_GID=998

# Pre-crea il gruppo docker con GID corretto PRIMA di installare docker.io
RUN groupadd -g ${DOCKER_GID} docker && \
    apt-get update && apt-get install -y docker.io && \
    usermod -aG docker $USERNAME

# Switch to non-root user
USER $USERNAME

# Set bash as default shell
RUN sudo chsh -s /bin/bash
