# hurricane-harm-herald
### Group Team Challenge 2023
*Owen Allemang, Lisanne Blok, Ruari Marshall-Hawkes, Orlando Timmerman, Peisong Zheng*

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

The destruction caused by hurricanes has huge social and economic impact. Between 1980-2008, 57\% of economic losses from worldwide natural disasters stem from storms making landfall across continental US and the Caribbean (NA). Hurricanes comprise the bulk of these. In continental US alone, 206 hurricane landfalls between 1900 and 2017 caused on average over \$20 billion in annual damage (Normalised from the 2018 to the 2022 dollar value). Development in vulnerable areas driven by population growth and increasing wealth increases average year-on-year losses. In addition, anthropogenic global warming correlates with the fractional proportion of high-intensity extreme weather events globally, including hurricanes and their associated environmental hazards such as storm surges.

ML is also increasingly used to predict the damage resulting from natural hazards. To date, hurricane damage prediction using ML has been limited, and damage prediction largely depends on probabilistic models of the response of physical building components to environmental stressors. These methods generally rely on time-intensive in-situ surveys and case studies of specific extreme weather events, which limits their ability to generalise. 

**Hurricane Harm Herald (3H)** uses a novel multimodal machine learning approach to predict the damage extent to buildings in response to forecasted weather features associated with major hurricanes (categories 3 to 5, as classified by the Saffir-Simpson scale). The tool uses openly accessible datasets to produce a building-level damage forecast map for regions of NA presently at risk of hurricanes. It is hoped that this may be made available for community-level decision making to increase the long-term resilience of neighbourhoods, prepare defences in response to forecasts of imminent storms, and provide preliminary direction for rescue workers following events for which damage assessments are not immediately available. The tool may also be useful to inform insurance policy. 

## Documentation

The `notebooks` folder contains interactive walk-throughs of data loading and visualisation, and model training and testing. The accompanying written report is in progress and will be uploaded to the repository once complete.

## Contributing

There is currently no opportunity to contribute to the project since it forms an assessed part of the AI4ER MRes year.

## License

This project falls under the MIT license.

## Acknowledgements

The team would like to thank Robert Muir-Wood for proposing the project, and Dominic Orchard, Grace Beaney-Colverd, and Luke Cullen for their support and expert guidance.


---

## Loading the dataset

### Getting the datasets
#### xBD
Download the dataset from https://xview2.org/ (you will need to create an account)
and put the files in `./data/datasets/XBD_data`.  
<i>Note: </i> The uncompressed data is about 130GB.
