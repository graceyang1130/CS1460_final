#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=64G
# Specify a partition
##SBATCH -p batch
#SBATCH -p gpu --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Pipeline
# Specify an output file
#SBATCH -o out/pipeline.out
#SBATCH -e out/pipeline.out

source ~/cs1430-venv/bin/activate
mkdir -p out/vis_pipeline
python pipeline.py

