#!/bin/bash

module load create
module load cuda
source $CREATE_HOME/av/helios/12.4/bin/setupEnv
python -m pip install --user pybind11==2.10.3
# mkdir build
# cd build
# CC=gcc CXX=g++ cmake ..
