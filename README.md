# GeoGoat

## Instructions
1. Start with running `download.sh` to download all data. Note that ideally all this should be done on Oscar or a similar system, because it uses special Oscar directories like `~/scratch` and is a very large amount of data and compute. It can be run with `sbatch download.sh`.
2. Run (`sbatch`) all the model training: `climate.sh`, `country.sh`, `continent.sh`, `shadow.sh`.
3. Train the ensemble model for the full pipeline with `sbatch pipeline.sh`. This will output example images of the full pipeline in `out/vis_pipeline/`.
4. Upload any JPG image to `demo/upload.jpg` and run `sbatch demo.sh` to see the prediction on this image in `out/demo.jpg`.

## Pipeline
The full GeoGoat pipeline takes predictions for all the different models and combines them into an actual latitude/longitude prediction.

Each model outputs a map of the world with 0.5x0.5 degree cells, and assigns a probability to each cell. The full pipeline normalizes the prediction for each model, weights them, and adds the predictions together. It then takes the cell with the highest value and outputs that as the final prediction. (There is also a little random noise added to avoid bias like always picking northwest in case of a tie). The relative weights of the different models and the random noise are not equal: they are trained as an ensemble model on a small sample of images (100).

## Climate model
This is a model that takes an image and outputs a probability distribution over the 30 different Köppen climate types plus an extra ocean type.

Since there is no large dataset we know of that has images labeled by Köppen climate type, we use the main Flickr geotagged image dataset and then map each image to a climate type using its coordinates and a separate dataset from [Beck et al. (2023)](https://www.gloh2o.org/koppen/). Most of the code is contained in `models.climate.py`. The model achieves a crossentropy loss of about 2.23 and a validation accuracy of about 28%. 28% accuracy sounds quite bad, but given that there are 31 climate types, this is providing a very useful signal over the 3.2% baseline if combined with other models.

Besides the standard training of a convolutional neural network (starting with a pretrained resnet model), the key piece of the climate model is the mapping from coordinates to climate, effectively bootstrapping a very large image/climate dataset from the existing Flickr dataset. This code is contained in the `coords_to_climate` function on line 23 of `models/climate.py`.

We also have a more coarse-grained climate model which groups the 30 Köppen climate types into just 5 broader groups. This model achieves a crossentropy loss of 1.04 and a validation accuracy of 62%.
