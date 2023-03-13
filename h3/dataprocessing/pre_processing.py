import numpy as np
import pandas as pd

import os
import rasterio as rio
# import cv2
# import scipy.ndimage
# from os.path import exists

from tqdm import tqdm
# from threading import Thread
# from h3 import logger
# from h3.constants import DMG_CLASSES_DICT
from h3.utils.directories import get_metadata_pickle_dir, get_xbd_hlabel_dir, \
    get_xbd_dir, get_data_dir
from h3.dataprocessing.extract_metadata import load_and_save_df
from h3.dataprocessing.crop_images_img import crop_images


def image_loading(polygons_df, zoom_levels: list, pixel_num: int,
                  zoomdir_dict: dict):
    """Loads images and crops them based on the required zoom levels
    and the required imagery input pixel size for the model.

    Parameters
    ----------
    polygons_df : geopandas dataframe
        Pandas dataframe containing metadata information about the pre-event
        polygoms, combined with the damage class from the post-event data.
        The reference system is "xy" referring to the corresponding image file.
    zoom_levels : list
        list containing all required zoom levels as integers.
    pixel_num : int
        Value of the sides of the squared cropped image as input for the model.
    zoomdir_dict : dict
        Dictionary containing all filepaths for the zoom directories with the
        values of the zoom level.
    """
    polygons_df["index"] = np.arange(len(polygons_df))
    filtered_df = polygons_df[["image_name", "json_link"]].drop_duplicates(
        subset=['image_name'])
    for json_path in tqdm(filtered_df["json_link"]):
        tif_name = json_path.replace("json", "tif")
        image_path = os.path.join(json_path, tif_name).replace("labels",
                                                               "images")
        # image name is in the final folder so index with -1
        name_img = os.path.basename(json_path).replace("json", "png")
        image_pol_df = polygons_df.query('image_name == @name_img')
        polygons_image = image_pol_df[["geometry", "index"]]
        with rio.open(image_path) as img:
            image_size = 1024
            for building in range(len(polygons_image)):
                polygon_for_img = polygons_image.iloc[building]["geometry"]
                polygon_num = polygons_image.iloc[building]["index"]

                for zoom_level in zoom_levels:
                    zoom_dir = zoomdir_dict[zoom_level]
                    img_num = str(polygon_num) + ".png"
                    output_path = os.path.join(zoom_dir, (img_num))
                    crop_images(img, polygon_for_img, zoom_level, pixel_num,
                                image_size, output_path)


def main():
    zoom_levels = [1, 2, 4, 0.5]
    pixel_num = 224

    # TODO: look/fix the geotiffs.old and all
    output_dir = get_metadata_pickle_dir()
    data_dir = get_data_dir()
    xbd_dir = get_xbd_dir()
    # xbd_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_geotiffs"
    # data_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_output/images"
    # output_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_output"

    # hold_filepath = get_xbd_hlabel_dir()
    hold_filepath = os.path.join(xbd_dir, "geotiffs", "hold", "labels")
    tier1_filepath = os.path.join(xbd_dir, "geotiffs", "tier1", "labels")
    tier3_filepath = os.path.join(xbd_dir, "geotiffs", "tier3", "labels")
    test_filepath = os.path.join(xbd_dir, "geotiffs", "test", "labels")

    filepaths_dict = dict.fromkeys([hold_filepath, tier1_filepath,
                                    tier3_filepath, test_filepath])

    # use df_pre_post_hurr_ll (longitude & latitude) for environmental factors
    # use df_pre_post_hurr_xy for image cropping
    df_pre_post_hurr_xy, df_pre_post_hurr_ll = load_and_save_df(
        filepaths_dict, output_dir)

    # where to save zoomed and cropped images
    save_dir_path = os.path.join(get_processed_data_dir(), "processed_xbd", "geotiffs_zoom", "images")

    zoomdir_dict = {}
    for zoom_num in zoom_levels:
        zoom_dir = "zoom_" + str(zoom_num)
        zoom_path = os.path.join(data_dir, save_dir_path, zoom_dir)
        zoomdir_dict[zoom_num] = zoom_path
        if not os.path.exists(zoom_path):
            os.makedirs(zoom_path)
    image_loading(df_pre_post_hurr_xy, zoom_levels, pixel_num, zoomdir_dict)


if __name__ == '__main__':
    main()
