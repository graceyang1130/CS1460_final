#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --mem=32G
# Specify a partition
#SBATCH -p batch
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Mapping
# Specify an output file
#SBATCH -o out/data-mapping.out
#SBATCH -e out/data-mapping.out

python data-mapping/flickr_map.py
