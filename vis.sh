#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --mem=32G
# Specify a partition
#SBATCH -p batch
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Vis
# Specify an output file
#SBATCH -o out/vis.out
#SBATCH -e out/vis.out

source ~/cs1430-venv/bin/activate
mkdir -p ~/scratch/GeoGoat/vis
python vis.py

