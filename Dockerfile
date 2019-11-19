FROM pytorch/pytorch:0.4_cuda9_cudnn7
COPY . /home/
WORKDIR /home/
RUN apt-get update && apt-get install git
