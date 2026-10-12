#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --mem=110G
# Specify a partition
#SBATCH -p batch
#SBATCH -N 1
# Specify a job name
#SBATCH -J GG_Download
# Specify an output file
#SBATCH -o out/download.out
#SBATCH -e out/download.out

mkdir -p ~/scratch/GeoGoat
curl -L -o ~/scratch/GeoGoat/large-dataset-of-geotagged-images.zip https://www.kaggle.com/api/v1/datasets/download/habedi/large-dataset-of-geotagged-images
cd ~/scratch/GeoGoat
rm -rf indices labels
unzip large-dataset-of-geotagged-images.zip
rm large-dataset-of-geotagged-images.zip

mkdir -p ~/scratch/GeoGoat/shapefiles
cd ~/scratch/GeoGoat/shapefiles
wget https://github.com/nvkelso/natural-earth-vector/raw/master/110m_cultural/ne_110m_admin_0_countries.shp
wget https://github.com/nvkelso/natural-earth-vector/raw/master/110m_cultural/ne_110m_admin_0_countries.shx
wget https://github.com/nvkelso/natural-earth-vector/raw/master/110m_cultural/ne_110m_admin_0_countries.dbf
wget https://github.com/nvkelso/natural-earth-vector/raw/master/110m_cultural/ne_110m_admin_0_countries.prj

mkdir -p ~/scratch/GeoGoat/koppen
cd ~/scratch/GeoGoat/koppen
wget https://figshare.com/ndownloader/files/45057352
unzip 45057352
rm 45057352
rm -rf 1901_1930 1931_1960 1961_1990 2041_2070 2071_2099
cd 1991_2020
rm koppen_geiger_0p00833333.tif koppen_geiger_0p1.tif
