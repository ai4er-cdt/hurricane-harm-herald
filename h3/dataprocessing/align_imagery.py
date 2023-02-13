import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import geopandas as gpd

path_to_folder = "/Users/Lisanne/Library/CloudStorage/GoogleDrive-lisanneblok@gmail.com/My Drive/ai4er/python/hurricane/hurricane-harm-herald/"
os.chdir(path_to_folder +'data/datasets/EFs')
df_points_polygons = pd.read_pickle("metadata_posthurr_points_polygons_lnglat_xy.pkl")

print(df_points_polygons['geometry'][0])