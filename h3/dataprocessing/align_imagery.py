import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fnmatch
import os
import geopandas as gpd

path_to_folder = "/Users/Lisanne/Library/CloudStorage/GoogleDrive-lisanneblok\
@gmail.com/My Drive/ai4er/python/hurricane/hurricane-harm-herald/"

os.chdir(path_to_folder)
# os.chdir(path_to_folder +'data/datasets/EFs')
# df_points_polygons = pd.read_pickle(
# "metadata_posthurr_points_polygons_lnglat_xy.pkl")
# print(df_points_polygons['geometry'][0])


# extract pre-event hurricane imagery
def extract_files(files, search_criteria):
    """Filters all json label files and returns a list of post-event files
    for hurricanes

    Args:
        files (list): list of json files in the label directory
        search_criteria (str): could filter out hurricanes, post-event imagery
        in json format
            i.e. input "hurricane*pre*json"

    Returns:
        list: list of filtered files for corresponding criteria
    """
    list_of_files = []
    for f in files:
        if fnmatch.fnmatch(f, search_criteria):
            list_of_files.append(f)
    return list_of_files


label_path = os.chdir(path_to_folder +
                      "data/datasets/geotiffs.old/hold/labels")
print(label_path)
directory_files = os.listdir()
full_post_hurr_json_files = extract_files(
    directory_files, "hurricane*post*.json")
print(os.getcwd())
