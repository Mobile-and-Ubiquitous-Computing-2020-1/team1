#!/usr/bin/env bash

# run it in $PROJECT_HOME/server

if [ ! -d "checkpoints" ]
then
  mkdir checkpoints
fi

cd checkpoints

if [ ! -f "resnet50.tar.gz" ]
then
  wget https://storage.googleapis.com/cloud-tpu-checkpoints/resnet/resnet50.tar.gz
fi

if [ ! -d "resnet50" ]
then
  tar -xvf resnet50.tar.gz

  mv home/hongkuny/hongkuny_keras_resnet50_gpu_8_fp32_eager_graph_cfit/checkpoints \
    resnet50

  rm -rf home
fi
