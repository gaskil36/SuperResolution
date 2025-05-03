# Applying Super-Resolution to Sentinel 2 Imagery
### Alexander Vu and Ben Gaskill

## Description:
The goal of our project is to research, adapt, and fine-tune super-resolution deep learning frameworks to resample Sentinel-2 imagery from its native resolution of 10 meters up to 1.5 meters per pixel. Our main focus is on the WorldStrat super-resolution model, which we tested using multi-temporal stacks of input imagery for 6 selected sites across Zambia. 

Please refer to ![Description.ipynb](https://github.com/gaskil36/superresolution/blob/main/Notebooks/description.ipynb) for the full description of our project.

## Visualization of Data and Results:  
Please refer to the ![SuperResolutionVisualizations]() notebook for dynamic visualizations (as well as our ![presentation slides](https://docs.google.com/presentation/d/1NXxHIwHK3bESZhNmiMa6fwTGc-tXAnu_QI6w-GREb9Q/edit#slide=id.p). Below is a static visualization of our inputs and outputs.  
___
![Selected Sites](Resources/SelectedSites.png)
___
### Input 1: Sentinel 2 (10 meter resolution, Site 0)  
![sentinel_full.png](Resources/sentinel_full.png)  
___
### Output 1: Super Resolution (1.5 meter resolution, Site 0, Per-Band Normalization)  
![per_band_full.png](Resources/per_band_full.png)
#### A closer look:  
#### Sentinel 2  
![per_band_sentinel.png](Resources/per_band_sentinel.png)  
#### Super Resolution  
![sentinel_zoom.png](Resources/per_band_zoom.png)  
#### Planet  
![site_0_planet.png](Resources/site_0_planet.png)  
___  
### Input 2: Sentinel 2 (1.5 meter resolution, Site 1, Cross-Band Normalization)
![site1_sentinel_full.png](Resources/site1_sentinel_full.png)
### Output 2: Super Resolution (1.5 meter resolution, Site 1, Cross-Band Normalization)  
![cross_band_full.png](Resources/cross_band_full.png)  
### A closer look:  
#### Sentinel 2  
![cross_band_sentinel.png](Resources/cross_band_sentinel.png)  
#### Super Resolution  
![cross_band_zoom.png](Resources/cross_band_zoom.png)  
#### Planet  
![site_1_planet.png](Resources/site1_planet.png)  
