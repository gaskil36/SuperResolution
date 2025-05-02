# Applying Super-Resolution to Sentinel 2 Imagery
### Alexander Vu and Ben Gaskill

## Description:
The goal of our project is to research, adapt, and fine-tune super-resolution deep learning frameworks to resample Sentinel-2 imagery from its native resolution of 10 meters up to 1.5 meters per pixel. Our main focus is on the WorldStrat super-resolution model, which we tested using multi-temporal stacks of input imagery for 6 selected sites across Zambia. The model accepts a multi-temporal stack of 8 images, along with all 12 bands of Sentinel 2 per study site. However, due to time constraints and Google Colab processing limits, we ran the inference pipeline only for Site 0. The model outputs the original 10-meter bands upscaled to 1.5 meters, namely the red, blue, and green bands.

We developed an imagery acquisition and preprocessing script and fed the input images into our inference pipeline. To assess the performance of the model, the super-resolution outputs were compared against the original Sentinel 2 images, as well as high-resolution Planet imagery and high-resolution imagery located at the same test site. We hope that our model can be utilized by the Agricultural Impacts Research Group to serve as a complement to paid high-resolution imagery such as Planet.  

## Division of Labor:
Ben developed a data acquisition and pre-processing pipeline for Sentinel 2 imagery. This notebook connects to Google Earth Engine and samples 6 bounding boxes within Zambia. 3 samples represent mixed urban areas, and 3 samples represent rural areas. By sampling geographically diverse areas, we hope to capture finer-grain details for our model to perform more effectively. The script processes cloud-free composites for 2024 by selecting imagery with less than 5 percent cloud coverage and masking cloudy pixels with the Scene Classification Layer. The script also uses Sentinel-2 footprints to ensure that each study area falls fully within one footprint and that there is valid data for the entire extent. The layers are then exported as GeoTiff files. Ben also worked on modifying and improving the outputs of the inference pipeline, namely the testing of different normalization strategies, creating Cloud-Optimized Geotiffs, and merging the chips into one final image. He developed the visualization notebook, which displays the results in leaflet maps.

Alex tested the WorldStrat super-resolution model using the pre-existing data in order to understand the input requirements. He also developed the inference pipeline, which chips the input images, adapts them to the model’s input, applies normalization, and builds the full super-resolution image.  

## Key Findings:  
We have found that the model performs reasonably well on our first study site. Most of the time spent on the project went into development and debugging, especially with applying normalization to the output. Multiple iterations of the image acquisition and processing pipeline notebooks were created, as well as many versions of the Super-Resolved output.

We decided to include 2 versions of the final output, in which one is normalized per-band and the other is normalized across the bands. We felt it useful to show both results, highlighting the strengths and weaknesses of each.

## Visualization of Data:  
Please refer to the ![SuperResolutionVisualizations]() notebook for dynamic visualizations (as well as our ![presentation slides](https://docs.google.com/presentation/d/1NXxHIwHK3bESZhNmiMa6fwTGc-tXAnu_QI6w-GREb9Q/edit#slide=id.p). Below is a static visualization of our inputs and outputs.  
![Selected Sites](Resources/SelectedSites.png)



