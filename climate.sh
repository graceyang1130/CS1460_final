#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=64G
# Specify a partition
##SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Climate
# Specify an output file
#SBATCH -o out/climate.out
#SBATCH -e out/climate.out

source ~/cs1430-venv/bin/activate
mkdir -p ~/scratch/GeoGoat/indices
mkdir -p ~/scratch/GeoGoat/labels
mkdir -p ~/scratch/GeoGoat/images
mkdir -p checkpoints
python models/climate.py
python models/climate2.py

