#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=64G
# Specify a partition
##SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Country
# Specify an output file
#SBATCH -o out/country.out
#SBATCH -e out/country.out

source /users/kjiang32/Desktop/GeoGoat/.venv/bin/activate
mkdir -p ~/scratch/GeoGoat/indices
mkdir -p ~/scratch/GeoGoat/labels
mkdir -p ~/scratch/GeoGoat/images
mkdir -p checkpoints
python models/country.py

