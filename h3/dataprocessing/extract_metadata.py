import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from shapely import wkt
import os
import fnmatch

import geopandas as gpd

from h3.utils.directories import get_xbd_dir
from h3.utils.directories import get_data_dir
from h3.utils.directories import get_xbd_hlabel_dir

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
def filter_files(files: list, filepath: str, search_criteria: str):
    """Filter all json label files and returns a list of post-event files for
     hurricanes.

    Parameters
    -----
        files : list
            list of json files in the label directory
        search_criteria : str
            filter out hurricanes, post-event imagery in json format
            i.e. input "hurricane*pre*json", with *'s as wildcard.

    Returns
    ------
        list 
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
    ------
        building : object
            polygon information in shapely coordinates.

    Returns
    -----
        object
            centroid point of polygon.
    """
    building_polygon = building["wkt"]
    building_coordinates = wkt.loads(building_polygon).centroid.wkt
    return building_coordinates


def extract_polygon(building):
    """Extract polygon coordinate reference system information.

    Parameters
    -----
        building : object
            polygon shapely coordinates.

    Returns
    -----
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
    -----
        json_link : path
            path to json file containing location and metadata.
        classes_dict : dict
            dictionary mapping between damage classes (str) and
            damage numbers (int).
        crs : str
            coordinate reference system to put as geometry in geodataframe.
        event_type : str
            post or pre event json files to filter out.

    Return:
        Geodataframe
            contains polygons of json file, corresponding metadata.
    """
    json_file = open(json_link)
    json_data = json.load(json_file)
    meta_data = json_data["metadata"]
    disaster_type = meta_data["disaster"]
    image_name = meta_data["img_name"]
    capture_date = meta_data["capture_date"]

    # for plotting on maps and finding environmental factors, use lng lat
    coordinates_lnlat = json_data["features"]["lng_lat"]
    # for coordination with imagery, use xy coordinates
    coordinates_xy = json_data["features"]["xy"]

    damage_location = []

    def damage(building_coordinates, event_type: str):
        """Extract damage class and maps it to a damage number according to the
        joint damage scale which is layed out in the classes dictionary. It
        returns a damage number. This only works for post-event json files.

        Parameters
        -----
        building_coordinates : object
            polygon coordinates of the building
        event_type : str
            pre-event or post-event tells you whether the geodataframe contains
            a column for damage class.

        Returns
        -------
        int
            damage number, based on the damage class from the json files. From
            CLASSES_DICT.
        """
        if event_type == "pre":
            damage_num = float('NaN')
        else:
            damage_class = building_coordinates["properties"]["subtype"]
            damage_num = CLASSES_DICT[damage_class]
        return damage_num

    for (building_lnglat, building_xy) in zip(coordinates_lnlat,
                                              coordinates_xy):
        lnglat_point = extract_point(building_lnglat)
        lnglat_polygon = extract_polygon(building_lnglat)
        xy_point = extract_point(building_xy)
        xy_polygon = extract_polygon(building_xy)

        # arbitrary if taking from xy or long lat features
        damage_num = damage(building_lnglat, event_type)

        damage_location.append([
            lnglat_point, lnglat_polygon, xy_point,
            xy_polygon, damage_num, disaster_type, image_name, capture_date])

    if crs == "xy":
        df = gpd.GeoDataFrame(
            damage_location,
            columns=["point_lnglat", "polygon_lnglat", "point_xy", "geometry",
                     "damage_class", "disaster_name", "image_name",
                     "capture_date"])

    else:
        df = gpd.GeoDataFrame(
            damage_location,
            columns=["geometry", "polygon_lnglat", "point_xy", "polygon_xy",
                     "damage_class", "disaster_name", "image_name",
                     "capture_date"])
    df["capture_date"] = pd.to_datetime(df["capture_date"])
    df["geometry"] = df["geometry"].apply(wkt.loads)
    return df


def extract_damage_allfiles_separate(directory_files: list, filepath: str,
                                     crs: str, event: str):
    """
    Filters all label files for hurricanes, extracts the metadata,
    concatenates all files for post and pre images separately.

    Parameters
    ------
        directory_files : list
            .json files in xBD data hold folder to filter.
        filepath : str
            file path to the image files.
        crs : str
            coordinate reference system to put as geometry in geodataframe.
        event : str
            post or pre event json files to filter out.

    Returns
    -----
        geodataframe: two geodataframes with a summary of metadata for all
        hurricane events with labels.
    """
    if event == "pre":
        full_hurr_json_files = filter_files(directory_files, filepath,
                                            "hurricane*pre*.json")
    if event == "post":
        full_hurr_json_files = filter_files(directory_files, filepath,
                                            "hurricane*post*.json")

    dataframes_list = []

    if len(full_hurr_json_files) > 0:
        for file in full_hurr_json_files:
            loc_and_damage_df = extract_metadata(file, CLASSES_DICT,
                                                 crs, event)
            dataframes_list.append(loc_and_damage_df)
            rdf = gpd.GeoDataFrame(pd.concat(dataframes_list,
                                             ignore_index=True))
    return rdf


def extract_damage_allfiles_ensemble(directory_files: list, filepath: str,
                                     crs: str):
    """
    Filters all pre and post label files for hurricanes, extracts the metadata
    from the post and pre json files. Takes damage information from post and
    adds that to the pre-event metadata dataframe.

    Parameters
    -----
        directory_files : list
            .json files in xBD data hold folder to filter.
        filepath : str
            file path to the image files.
        crs : str
            coordinate reference system to put as geometry in
            geodataframe.

    Returns
    -----
        geodataframe:
            geodataframes with a summary of metadata for all
            pre-event hurricane events with post-event labels.
    """
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

    full_pre_hurr_json_files = filter_files(directory_files, filepath,
                                            "hurricane*pre*.json")
    pre_dataframes_list = []
    for pre_json_name in full_pre_hurr_json_files:
        post_json_name = pre_json_name.replace("pre", "post")

        pre_metadata = extract_metadata(
            pre_json_name, CLASSES_DICT, crs, "pre")
        post_metadata = extract_metadata(
            post_json_name, CLASSES_DICT, crs, "post")

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
        polygons_pre = pre_metadata.merge(merge_post_metadata, on="match_num")
        pre_dataframes_list.append(polygons_pre)
    pre_rdf = gpd.GeoDataFrame(pd.concat(pre_dataframes_list,
                                         ignore_index=True))
    return pre_rdf


def load_and_save_df():
    """
    Loads the json label files for all hurricanes in the "hold" section of the
    xBD data, extracts the points and polygons in both xy coordinates,
    referring to the corresponding imagery file, and the longitude and
    latitude.

    Parameters
    -----
    -

    Returns
    -----
        Geodataframe
            all metadata and locations in a geodataframe that is
            saved in the data/datasets/EFs directory. Choose to return the gdf
            with long-lat coordinate system and pre polygons with post damage
            as this is most useful for choosing the EFs.
    """
    data_dir = get_data_dir()
    # xbd_dir = get_xbd_dir()
    # label path
    filepath = get_xbd_hlabel_dir()  # TODO: look/fix the geotiffs.old and all
    # filepath = os.path.join(xbd_dir, labels_path, "")
    fulldirectory_files = [os.path.join(filepath, file)
                           for file in os.listdir(filepath)]

    df_points_post_hurr = extract_damage_allfiles_separate(
        fulldirectory_files, filepath, "xy", "pre")
    path_save_post = os.path.join(
        data_dir,
        "datasets/processed_data/metadata_pickle",
        "pre_polygon.pkl")
    df_points_post_hurr.to_pickle(path_save_post)

    df_pre_post_hurr_xy = extract_damage_allfiles_ensemble(
        fulldirectory_files, filepath, "xy")
    path_save_pre = os.path.join(
        data_dir,
        "datasets/processed_data/metadata_pickle",
        "xy_pre_pol_post_damage.pkl")
    df_pre_post_hurr_xy.to_pickle(path_save_pre)

    df_pre_post_hurr_ll = extract_damage_allfiles_ensemble(
        fulldirectory_files, filepath, "lng_lat")
    path_save_pre_longlat = os.path.join(
        data_dir,
        "datasets/processed_data/metadata_pickle",
        "lnglat_pre_pol_post_damage.pkl")
    df_pre_post_hurr_ll.to_pickle(path_save_pre_longlat)

    return df_pre_post_hurr_ll


def main():
    load_and_save_df()


if __name__ == "__main__":
    main()
