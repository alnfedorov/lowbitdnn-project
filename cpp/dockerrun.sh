#!/bin/bash
sudo docker build --rm --tag=fedorov .
sudo docker run -it -d --runtime=nvidia --ipc=host --privileged=true -v /home/fedorov:/home/fedorov -p 22222:22 --name=fedorov fedorov