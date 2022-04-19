#!/bin/bash
source /home/ubuntu/anaconda3/bin/activate Openpcdet
export PATH=/usr/local/cuda-11.3/bin:$PATH
echo $PATH
cd $1 && python cmake_setup.py ${@:2}