# Useses the GPU but works only on server 1
FROM tensorflow/tensorflow:1.13.1-gpu-py3

# this workaround is to avoid the "public key is not available" error:
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get upgrade -y

RUN pip3 install --upgrade pip==20.2

WORKDIR /tmp/

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

# RUN apt-get update && apt-get install -y htop git tmux

ARG USERNAME=ruggeri
ARG USER_UID=2565
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# change default shell to bash for new user
RUN sudo chsh -s /bin/bash

# change default tmux shell to bash for new user
# RUN echo "set-option -g default-shell /bin/bash" > ~/.tmux.conf