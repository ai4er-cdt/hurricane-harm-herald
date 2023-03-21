# hurricane-harm-herald
### Group Team Challenge 2023
*Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![docs](https://github.com/ai4er-cdt/hurricane-harm-herald/actions/workflows/sphinx.yml/badge.svg)](https://ai4er-cdt.github.io/hurricane-harm-herald/)

The destruction caused by hurricanes has huge social and economic impact. Between 1980-2008, 57\% of economic losses from worldwide natural disasters stem from storms making landfall across continental US and the Caribbean. Hurricanes comprise the bulk of these. In continental US alone, 206 hurricane landfalls between 1900 and 2017 caused on average over \$20 billion in annual damage (Normalised from the 2018 to the 2022 dollar value). Development in vulnerable areas driven by population growth and increasing wealth increases average year-on-year losses. In addition, anthropogenic global warming correlates with the fractional proportion of high-intensity extreme weather events globally, including hurricanes and their associated environmental hazards such as storm surges.

To date hurricane damage prediction largely depends on probabilistic models of the response of physical building components to environmental stressors. These methods generally rely on time-intensive in-situ surveys and case studies of specific extreme weather events, which limits their ability to generalise. While use of ML in building damage classification is well-established thanks largely to the 2019 xView2 [click here](https://xview2.org/) Challenge, damage prediction using ML is limited. 

containing over 700,000 building annotations and labels. 
**Hurricane Harm Herald (3H)** uses a novel multimodal machine learning approach to predict the damage extent to buildings in response to forecasted weather features associated with major hurricanes (categories 3 to 5, as classified by the Saffir-Simpson scale). The tool uses openly accessible datasets to produce a building-level damage forecast map for regions of NA presently at risk of hurricanes. It is hoped that this may be made available for community-level decision making to increase the long-term resilience of neighbourhoods, prepare defences in response to forecasts of imminent storms, and provide preliminary direction for rescue workers following events for which damage assessments are not immediately available. The tool may also be useful to inform insurance policy. 

## Documentation

Functions documentation is available (https://ai4er-cdt.github.io/hurricane-harm-herald/)

The `notebooks` folder contains interactive walk-throughs of data loading and visualisation, and model training and testing. The accompanying written report is in progress and will be uploaded to the repository once complete.

## Contributing

There is currently no opportunity to contribute to the project since it forms an assessed part of the AI4ER MRes year.

## License

This project falls under the MIT license.

## Acknowledgements

The team would like to thank Robert Muir-Wood for proposing the project, and Dominic Orchard, Grace Beaney-Colverd, and Luke Cullen for their support and expert guidance.


---

## Environment setup

It supports `python>=3.10`.
We recommend using `conda` or [`mamba`](https://mamba.readthedocs.io/en/latest/installation.html) to install
the dependencies.  
You can install with `pip`. However, since we are dependent on `rasterio`, that requires C compiled code,
we cannot guarantee the installation with `pip` on Windows. 

### CONDA | CUDA
```shell
conda env create -f environment_gpu.yml
```

### CONDA | no CUDA
```shell
conda env create -f environment_no_gpu.yml
```

### PIP
You may want to install it in a virtual environment.

```shell
pip install -e .
```

If you want to only install the requirements.
```shell
pip install -r requirement.txt
```


## Loading the datasets

There are several datasets used to test and train our model. The following sections describe how to download each dataset. Due to the size of data, it's recommended to use a remote storage service, for example a [Google Drive](https://www.google.co.uk/intl/en-GB/drive/).

The overall file structure should be as follows:

```
 ├── data
 │      ├── datasets
 │      │      └── xBD_data
 │      │      └── DEM_data
 │      │      └── storm_surge_flood_data
 │      │      └── DEM_data
 │      │      └── weather_data
 │      │           └── ecmwf_era5
 │      │           └── noaa_best_track
 │      │           └── global_isd
 │      │           └── noaa_best_track
```

### xBD pre- and post-event satellite damage-annotated imagery 
[xBD](https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.pdf) is the dataset used in the xView2 challenge, providing pre- and post-event RGB satellite imagery with over 700,000 building polygons. 

Download the dataset from https://xview2.org/ (you will need to register for an account) and put the files in `./data/datasets/xBD_data`.  
<i>Note: </i> The uncompressed data is about 130GB.

After downloading, the data will be organised in the directories as follows:

`/data/datasets/xBD_data/geotiffs/`

```
 ├── tier1  
 │      ├── images  
 │      │      └── <image_id>.png  
 │      │      └── ...  
 │      └── labels  
 │             └── <image_id>.json  
 │             └── ...  
 ├── tier3
 │      ├── images  
 │      │      └── <image_id>.png  
 │      │      └── ...  
 │      └── labels  
 │             └── <image_id>.json  
 │             └── ...  
 ├── test  
 │      ├── images  
 │      │      └── <image_id>.png  
 │      │      └── ...  
 │      └── labels  
 │             └── <image_id>.json  
 │             └── ...  
 └── holdout  
        ├── images  
        │      └── <image_id>.png  
        │      └── ...  
        └── labels  
               └── <image_id>.json  
               └── ...  
```

### DEM 

To download the DEM files, you need an account here: https://urs.earthdata.nasa.gov/users/new/  
Once your account has been created, have your credentials on hand to input them when needed  
(The credentials will be stored in `./data/credentials.json`)


### Weather Data

Weather data from the Global Integrated Surface Dataset, NOAA HURDAT2 Best Track data, and ERA5-Land Reanalysis can be downloaded by running the `download_weather_data.ipynb` notebook. This will be downloaded in the following file structure:

```
 ├── weather_data
 │      ├── ecmwf_era5
 │      │      └── <image_id>.png
 │      │      └── ...
 │      └── noaa_best_track
 │      |      └── <image_id>.json
 │      |      └── ...
 │      ├── global_isd
 │             └── <image_id>.png
 │             └── ...
```



## Contributors

<a href="https://github.com/ai4er-cdt/hurricane-harm-herald/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ai4er-cdt/hurricane-harm-herald" />
</a>

Made with [contrib.rocks](https://contrib.rocks).


## License
This software is double_licensed.  
The main code is under MIT license, the SatMAE model is under the Attribution-NonCommercial 4.0 International.  
The SatMAE code is in `h3.models.SatMAE`, please look at the README there for more information.  
Additionally, the SatMAE code was modify to make it compatible with `numpy=1.24`
