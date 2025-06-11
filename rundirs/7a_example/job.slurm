#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --qos=prod
#SBATCH --partition=prodgpu
#SBATCH --job-name=7afoil
#SBATCH --account=AEDCA19682S2G
#SBATCH --export=ALL
#SBATCH --exclusive

# sym-link g2d.exe from the build directory here

./g2d.exe > out.txt
