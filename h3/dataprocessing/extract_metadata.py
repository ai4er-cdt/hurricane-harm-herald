import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from shapely import wkt
import os
import fnmatch

import rasterio as rio
from rasterio.plot import show
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from rasterio.mask import mask
import geopandas as gpd

# from h3.utils.directories import get_xbd_dir

path_to_folder = "/Users/Lisanne/Library/CloudStorage/GoogleDrive-\
lisanneblok@gmail.com/My Drive/ai4er/python/hurricane/hurricane-harm-herald/"

# Convert different damage classes (Joint Damage Scale) into integers
# NEED TO CONVERT TO INMUTABLE DICTIONARY
CLASSES_DICT = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4
}


# extract pre-event hurricane imagery
def filter_files(files: list, search_criteria: str):
    """Filter all json label files and returns a list of post-event files for
     hurricanes.

    Args:
        files (list): list of json files in the label directory
        search_criteria (str): could filter out hurricanes, post-event imagery
        in json format i.e. input "hurricane*pre*json", with *'s as wildcard.

    Returns:
        list: list of filtered files for corresponding criteria.
    """
    list_of_files = []
    for f in files:
        if fnmatch.fnmatch(f, search_criteria):
            list_of_files.append(f)
    return list_of_files


def extract_point(building):
    """Extract coordinate information from polygon and convert to a centroid
    point.

    Args:
        building (object): polygon information in shapely coordinates.

    Returns:
        object: centroid point of polygon.
    """
    building_polygon = building["wkt"]
    building_coordinates = wkt.loads(building_polygon).centroid.wkt
    return building_coordinates


def extract_polygon(building):
    """Extract polygon coordinate reference system information.

    Args:
        building (object): polygon shapely coordinates.

    Returns:
        object: polygon with spatial coordinate information.
    """
    building_polygon = building["wkt"]
    return building_polygon


def extract_metadata(json_link, CLASSES_DICT):
    """
    Extracts location in xy and long-lat format, gives damage name, class and
    date.

    Args:
        json_link (path): path to json file containing location and metadata.

    Return:
        Geodataframe: contains polygons of json file, corresponding metadata.
    """
    json_file = open(json_link)
    json_data = json.load(json_file)
    meta_data = json_data["metadata"]
    disaster_type = meta_data["disaster"]
    capture_date = meta_data["capture_date"]

    # for plotting on maps and finding environmental factors, use lng lat
    coordinates_lnlat = json_data["features"]["lng_lat"]
    # for coordination with imagery, use xy coordinates
    coordinates_xy = json_data["features"]["xy"]

    damage_location = []
    for (building_lnglat, building_xy) in zip(coordinates_lnlat,
                                              coordinates_xy):
        lnglat_point = extract_point(building_lnglat)
        lnglat_polygon = extract_polygon(building_lnglat)
        xy_point = extract_point(building_xy)
        xy_polygon = extract_polygon(building_xy)

        # arbitrary if taking from xy or long lat features
        damage_class = building_lnglat["properties"]["subtype"]
        damage_num = CLASSES_DICT[damage_class]

        damage_location.append([
            lnglat_point, lnglat_polygon, xy_point,
            xy_polygon, damage_num, disaster_type, capture_date])
    df = gpd.GeoDataFrame(
        damage_location,
        columns=["geometry", "polygon_lnglat", "point_xy", "polygon_xy",
                 "damage_class", "disaster_name", "capture_date"])
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return df


def extract_damage_allfiles(directory_files):
    """
    Filters all label files for hurricanes, extracts the metadata,
    concatenates all files.

    Args:
        directory_files (list): .json files in xBD data hold folder to filter.

    Returns:
        geodataframe: summary of metadata for all hurricane events with labels.
    """
    dataframes_list = []
    full_post_hurr_json_files = filter_files(directory_files,
                                             "hurricane*post*.json")
    for file in full_post_hurr_json_files:
        loc_and_damage_df = extract_metadata(file, CLASSES_DICT)
        dataframes_list.append(loc_and_damage_df)
        rdf = gpd.GeoDataFrame(pd.concat(dataframes_list, ignore_index=True))
    return rdf


def load_and_save_df() -> None:
    """
    Loads the json label files for all hurricanes in the "hold" section of the
    xBD data, extracts the points and polygons in both xy coordinates,
    referring to the corresponding imagery file, and the longitude and
    latitude.

    Args:

    Returns:
        Geodataframe: all metadata and locations in a geodataframe that is
        saved in the data/datasets/EFs directory.
    """

    # xbd_dir = get_xbd_dir()
    # os.path.join(xbd_dir)

    os.chdir(path_to_folder +
             "data/datasets/xbd_data/geotiffs.old/hold/labels")
    files = os.listdir()
    print(files)

    df_points_post_hurr = extract_damage_allfiles(files)

    # path_save = os.path.join(
    #    path_to_folder,
    #    "data/datasets/EFs/",
    #    "metadata_posthurr_points_polygons_lnglat_xy.pkl")
    # df_points_post_hurr.to_pickle(path_save)


def main():
    load_and_save_df()


if __name__ == "__main__":
    main()
