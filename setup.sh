#!/usr/bin/env bash
# assuming nvidia docker container with cuda 10.0 and cudnn 7.5
# BUILD ONLY INSIDE VENV!!!!!

touch ~/.no_auto_tmux
apt install -y libboost-dev cmake cmake-curses-gui htop git python3-pip libblas3 liblapack3 liblapack-dev libblas-dev
pip3 install numpy scipy opencv-python pyyaml numba mkl mkl-include setuptools cmake cffi typing ninja yacs cython matplotlib tqdm pytest
pip3 install pycocotools 

cd /home/
apt install -y wget unzip
wget https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

sed -i 's/#PermitRootLogin no/PermitRootLogin yes/g' /etc/ssh/sshd_config
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config

sed -i 's/PermitRootLogin no/PermitRootLogin yes/g' /etc/ssh/sshd_config
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/g' /etc/ssh/sshd_config
service ssh restart

cd /home/
git clone https://github.com/pytorch/pytorch.git
cd pytorch 
git pull
git checkout tags/v1.0.1
git submodule update --init --recursive

python3 setup.py bdist_wheel
cd dist
python3 -m pip install *.whl
