# Applying Super-Resolution to Sentinel 2 Imagery
### Alexander Vu and Ben Gaskill

## Project Overview:
The goal of our project is to research, adapt, and fine-tune multiple pre-existing super-resolution deep learning frameworks to resample Sentinel-2 imagery from its native resolution of 10 meters up to 1.5 meters per pixel. We will begin by testing and understanding the existing WorldStrat super-resolution model, including its architecture and input requirements. This model will be adapted to our needs and integrated with any additional models we find. Our focus is applying super-resolution to the 10 meter bands of Sentinel 2 imagery, namely the red, blue, green, and near infrared bands. We hope that our model can be utilized by the Agricultural Impacts Research Group to serve as a complement to paid high-resolution imagery such as Planet.

## Division of Labor: 
Ben is responsible for developing a data acquisition and pre-processing pipeline for Sentinel 2 imagery. This notebook connects to Google Earth Engine and samples 6 bounding boxes within Zambia. 3 samples represent mixed urban areas, and 3 samples represent rural areas. By sampling geographically diverse areas, we hope to capture finer-grain details for our model to perform more effectively. The script processes cloud-free composites for 2024 by selecting imagery with less than 5 percent cloud coverage, and masking cloudy pixels with the Scene Classification Layer. The layers are then exported as GeoTiff files, where the dimensions are clipped and equalized across samples to have an equal height and width. Finally, the script will process the data further (image chipping, etc) to meet the input requirements of our model.

Alex is responsible for testing the WorldStrat super-resolution model using the pre-existing data in order to understand the input requirements. He will also be responsible for the inference pipeline, which will take the chipped data, adapt it to the model’s input, and build the full super-resolution image. Then, the super-resolution image will be exported to a tiff file.

We are both responsible for researching additional super-resolution models, and will jointly work on adapting and fine-tuning them to create a custom super-resolution deep learning framework. We will then compare the performance of our chosen models on different geographies.
