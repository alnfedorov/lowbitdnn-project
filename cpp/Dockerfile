# Use an official nvidia-cuda-cudnn runtime as a parent image
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    libboost-dev \
    cmake \
    htop \
    git \
    wget \
    unzip \
    build-essential \
    openssh-server \
    python3.7-dev \
 && rm -rf /var/lib/apt/lists/*

RUN echo "root:2Science2" | chpasswd && \
    sed -i 's/#PermitRootLogin no/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 22 available to the world outside this container
EXPOSE 22
