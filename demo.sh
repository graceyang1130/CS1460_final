#!/bin/bash

#SBATCH --time=0:20:00
#SBATCH --mem=64G
# Specify a partition
##SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Demo
# Specify an output file
#SBATCH -o out/demo.out
#SBATCH -e out/demo.out

source ~/cs1430-venv/bin/activate
python demo/demo.py

