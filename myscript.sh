#!/bin/bash
#
# MyConda Wrapper Script
#
export PATH=/data/xiaoshua/anaconda37/bin:$PATH
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:/usr/local/cudnn8.0-10.2/lib64:/data/xiaoshua/anaconda37/lib/python3.7/site-packages/torch/lib:/data/xiaoshua/anaconda37/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda-10.2/lib64:/usr/local/cudnn8.0-10.2/lib64:/data/xiaoshua/anaconda37/lib/python3.7/site-packages/torch/lib:$LIBRARY_PATH
export CPATH=/usr/local/cuda-10.2/include:/usr/local/cudnn8.0-10.2/include:$CPATH

export CUDA_HOME=/usr/local/cuda-10.2

bash
