from __future__ import annotations

import os
import fnmatch
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely import wkt
from tqdm import tqdm

from typing import Literal
from h3.constants import DMG_CLASSES_DICT
from h3.utils.directories import get_metadata_pickle_dir, get_xbd_hlabel_dir, \
    get_xbd_dir


# extract pre-event hurricane imagery
def filter_files(files: list, filepath: str, search_criteria: str) -> list:
    """Filter all json label files and returns a list of post-event files for
     hurricanes.

    Parameters
    ----------
    files : list
        list of json files in the label directory
    filepath: str
        path to file, assisting in search criteria process
    search_criteria : str
        filter out hurricanes, post-event imagery in json format.
        i.e. input `hurricane*pre*json`, supports glob wildcard `*`

    Returns
    -------
    list:
        list of filtered files for corresponding criteria.
    """
    list_of_files = []
    search_path = os.path.join(filepath, search_criteria)

    for f in files:
        if fnmatch.fnmatch(f, search_path):
            list_of_files.append(f)
    return list_of_files


def extract_point(building):
    """Extract coordinate information from polygon and convert to a centroid
    point.

    Parameters
    ----------
    building : object
        polygon information in shapely coordinates.

    Returns
    -------
    object
        centroid point of polygon.
    """
    building_polygon = building["wkt"]
    building_coordinates = wkt.loads(building_polygon).centroid.wkt
    return building_coordinates


def extract_polygon(building):
    """Extract polygon coordinate reference system information.

    Parameters
    ----------
    building : object
        polygon shapely coordinates.

    Returns
    -------
    object
        polygon with spatial coordinate information.
    """
    building_polygon = building["wkt"]
    return building_polygon


def extract_metadata(json_link: str, CLASSES_DICT: dict, crs: str,
                     event_type: str):
    """
    Extracts location in xy and long-lat format, gives damage name, class and
    date.

    Parameters
    ----------
    json_link : path
        path to json file containing location and metadata.
    classes_dict : dict
        dictionary mapping between damage classes (str) and
        damage numbers (int).
    crs : str
        coordinate reference system to put as geometry in geodataframe.
    event_type : str
        post or pre event json files to filter out.

    Returns
    -------
    Geodataframe
        contains polygons of json file, corresponding metadata.
    """
    # json_file = open(json_link)
    # json_data = json.load(json_file)
    with open(json_link, 'r') as j:
        json_data = json.loads(j.read())

    meta_data = json_data["metadata"]
    disaster_type = meta_data["disaster"]
    image_name = meta_data["img_name"]
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
        if event_type == "pre":
            damage_num = np.NaN
        else:
            damage_class = building_lnglat["properties"]["subtype"]
            damage_num = CLASSES_DICT[damage_class]

        damage_location.append([
            lnglat_point, lnglat_polygon, xy_point,
            xy_polygon, damage_num, disaster_type, image_name, capture_date,
            json_link])
    if crs == "xy":
        df = gpd.GeoDataFrame(
            damage_location,
            columns=["point_lnglat", "polygon_lnglat", "point_xy", "geometry",
                     "damage_class", "disaster_name", "image_name",
                     "capture_date", "json_link"])

    else:
        df = gpd.GeoDataFrame(
            damage_location,
            columns=["geometry", "polygon_lnglat", "point_xy", "polygon_xy",
                     "damage_class", "disaster_name", "image_name",
                     "capture_date", "json_link"])
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return df


def extract_damage_allfiles_separate(filepaths_dict: dict,
                                     crs: str, event: Literal["pre", "post"]):
    """
    Filters all label files for hurricanes, extracts the metadata,
    concatenates all files for post and pre images separately.

    Parameters
    ----------
    directory_files : dict
        .json files in xBD data folder to filter organised by their folder.
        These files are a value for the holdout, tier1, tier3 and test folder
        as a key.
    crs : str
        coordinate reference system to put as geometry in geodataframe.
    event : str
        post or pre event json files to filter out.

    Returns
    -------
    geodataframe
        two geodataframes with a summary of metadata for all
        hurricane events with labels.
    """
    if event == "pre":
        search_criterium = "hurricane*pre*.json"
    if event == "post":
        search_criterium = "hurricane*post*.json"

    dataframes_list = []
    for directory in filepaths_dict:
        filepath_list = (filepaths_dict[directory])
        full_hurr_json_files = filter_files(filepath_list,
                                            directory,
                                            search_criterium)
        if len(full_hurr_json_files) > 0:
            for file in tqdm(full_hurr_json_files,
                             desc=f"Extracting metadata for {event}"
                             "hurricane"):
                loc_and_damage_df = extract_metadata(file, DMG_CLASSES_DICT,
                                                     crs, event)
                dataframes_list.append(loc_and_damage_df)
    rdf = gpd.GeoDataFrame(pd.concat(dataframes_list,
                                     ignore_index=True))
    return rdf


# check if polygons from pre and post overlap
def overlapping_polygons(geoms, p):
    """Checks if polygons from pre- and post-event imagery overlap.
    If they do, the damage class from post-event dataframe can be allocated
    to the pre-event polygon.

    Parameters
    ----------
    geoms : series
        post-event geodataframe geometry column containing the polygon
    p : object
        pre-event polygon extracted from geodataframe

    Returns
    -------
    series
        column in post-event dataframe containing which row number the
        post-event polygon matches with in the pre-event dataframe.
    """
    overlap = (geoms.intersection(p).area / p.area) > 0.7
    return pd.Series(overlap.index[overlap])


def extract_damage_allfiles_ensemble(
        filepaths_dict: dict,
        crs: str,
):
    """
    Filters all pre and post label files for hurricanes, extracts the metadata
    from the post and pre json files. Takes damage information from post and
    adds that to the pre-event metadata dataframe.

    Parameters
    ----------
     filepaths_dict : dict
        .json files in xBD data folder to filter organised by their folder.
        These files are a value for the holdout, tier1, tier3 and test folder
        as a key.
    crs : str
        coordinate reference system to put as geometry in geodataframe.

    Returns
    -------
    geodataframe
        geodataframes with a summary of metadata for all
        pre-event hurricane events with post-event labels.
    """
    full_pre_dataframes_list = []
    for directory in filepaths_dict:
        full_pre_hurr_json_files = filter_files(filepaths_dict[directory],
                                                directory,
                                                "hurricane*pre*.json")
        # pre_dataframes_list = []
        for pre_json_name in tqdm(full_pre_hurr_json_files,
                                  desc=f"Extracting metadata for pre event and post damage label hurricane"):
            post_json_name = pre_json_name.replace("pre", "post")

            pre_metadata = extract_metadata(
                pre_json_name, DMG_CLASSES_DICT, crs, "pre")
            post_metadata = extract_metadata(
                post_json_name, DMG_CLASSES_DICT, crs, "post")

            # which row matches with which?
            # post_metadata["match_num"] = pre_metadata.geometry.apply(
            # lambda x: overlapping_polygons(post_metadata, x))
            post_metadata["match_num"] = post_metadata.index
            merge_post_metadata = post_metadata[["damage_class", "match_num"]]
            pre_metadata["match_num"] = pre_metadata.index
            pre_metadata = pre_metadata.drop("damage_class", axis=1)
            # Assume order of polygons in pre and post json data is same
            # otherwise, use the overlapping_polygons function which indicates
            # polygonsare overlapping and are therefore pre and post pairs
            polygons_pre = pre_metadata.merge(merge_post_metadata,
                                              on="match_num")
            polygons_pre = polygons_pre.drop(["match_num"], axis=1)
            full_pre_dataframes_list.append(polygons_pre)
    pre_rdf = gpd.GeoDataFrame(pd.concat(full_pre_dataframes_list,
                                         ignore_index=True))
    return pre_rdf


def load_and_save_df(filepaths_dict: dict, output_dir: str, reload_pickle: bool = False):
    """
    Loads the json label files for all hurricanes in the "hold" section of the
    xBD data, extracts the points and polygons in both xy coordinates,
    referring to the corresponding imagery file, and the longitude and
    latitude.

    Parameters
    ----------
    filepaths_dict : dict
        pathnames in a dictionary for the holdout, tier1, tier3 and test
        folder.
    output_dir : str
    reload_pickle : bool, optional
        If True recreate the pickle files as if they did not exist. The default is False.

    Returns
    -------
    Geodataframe
        all metadata and locations in a geodataframe that is
        saved in the data/datasets/EFs directory. Choose to return the gdf
        with long-lat coordinate system and pre polygons with post damage
        as this is most useful for choosing the EFs.
    """
    # update filepaths_dictionary
    for filepath in filepaths_dict:
        directory_files = [os.path.join(filepath, file)
                           for file in os.listdir(filepath)]
        filepaths_dict[filepath] = directory_files

    path_save_post = os.path.join(output_dir, "pre_polygon.pkl")
    if not os.path.exists(path_save_post) or reload_pickle:
        df_points_post_hurr = extract_damage_allfiles_separate(
            filepaths_dict=filepaths_dict,
            crs="xy",
            event="pre"
        )
        df_points_post_hurr.to_pickle(path_save_post)
    else:
        df_points_post_hurr = pd.read_pickle(path_save_post)    # TODO: validate this

    path_save_pre = os.path.join(output_dir, "xy_pre_pol_post_damage.pkl")
    if not os.path.exists(path_save_pre) or reload_pickle:
        df_pre_post_hurr_xy = extract_damage_allfiles_ensemble(
            filepaths_dict=filepaths_dict,
            crs="xy"
        )
        df_pre_post_hurr_xy.to_pickle(path_save_pre)
    else:
        df_pre_post_hurr_xy = pd.read_pickle(path_save_pre)

    path_save_pre_longlat = os.path.join(output_dir, "lnglat_pre_pol_post_damage.pkl")
    if not os.path.exists(path_save_pre_longlat) or reload_pickle:
        df_pre_post_hurr_ll = extract_damage_allfiles_ensemble(
            filepaths_dict=filepaths_dict,
            crs="lng_lat"
        )
        df_pre_post_hurr_ll.to_pickle(path_save_pre_longlat)
    else:
        df_pre_post_hurr_ll = pd.read_pickle(path_save_pre_longlat)

    return df_pre_post_hurr_xy, df_pre_post_hurr_ll


def main():
    xbd_dir = get_xbd_dir()
    #xbd_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_geotiffs"
    output_dir = get_metadata_pickle_dir()
    #output_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_output"
    # hold_filepath = get_xbd_hlabel_dir()
    hold_filepath = os.path.join(xbd_dir, "geotiffs/hold/labels")
    tier1_filepath = os.path.join(xbd_dir, "geotiffs/tier1/labels")
    tier3_filepath = os.path.join(xbd_dir, "geotiffs/tier3/labels")
    test_filepath = os.path.join(xbd_dir, "geotiffs/test/labels")
    # filepath = os.path.join(xbd_dir, labels_path, "")
    filepaths_dict = dict.fromkeys([hold_filepath, tier1_filepath,
                                    tier3_filepath, test_filepath])
    df_pre_post_hurr_xy, df_pre_post_hurr_ll = load_and_save_df(filepaths_dict, output_dir)
    return df_pre_post_hurr_xy, df_pre_post_hurr_ll


if __name__ == "__main__":
    main()
