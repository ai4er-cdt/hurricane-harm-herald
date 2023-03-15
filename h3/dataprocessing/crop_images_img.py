import numpy as np
import pandas as pd

import os
import rasterio as rio
import cv2
import scipy.ndimage
from os.path import exists

from tqdm import tqdm
from threading import Thread

from h3 import logger
from h3.utils.directories import get_xbd_dir
from h3.utils.directories import get_data_dir
from h3.dataprocessing.extract_metadata import extract_damage_allfiles_ensemble


def extract_coords(row):
    return np.array(row).astype(np.int32)


def polygon_mask(img_array, polygon, im_size: int):
    """Fills up a polygon to create a mask of the building.

    Parameters
    ----------
    img : dataset object
        pre-event image file aligning with the building polygon.
    polygon : object
        polygon containing the outline of the building.
    im_size : int
        Value of the sides of the squared input image.

    Returns
    -------
    numpy array
        filled mask of the building.
    """
    image_size = (im_size, im_size)

    empty_mask = np.zeros(image_size, np.uint8)
    poly_coords = [extract_coords(polygon.exterior.coords)]
    image_mask = cv2.fillPoly(empty_mask, poly_coords, 1)
    return image_mask


def mask_to_bb(Y):
    """Takes a filled building mask and converts it into a
    rectangular bounding box.

    Parameters
    ----------
    Y : numpy array
        The filled polgyon mask from the polygon_mask function.

    Returns
    -------
    numpy array
        Array of the rectangular bounding box as a mask.
    """
    nx, ny = np.nonzero(Y)
    # check if polygon is empty
    if len(nx) == 0 or len(ny) == 0:
        return np.zeros(4, dtype=np.float32)
    top_y = np.min(ny)
    bottom_y = np.max(ny)
    left_x = np.min(nx)
    right_x = np.max(nx)
    return np.array([left_x, top_y, right_x, bottom_y], dtype=np.int64)


def crop_images(image_array, img_metadata, polygon_df, zoom_level: int, pixel_num: int,
                im_size: int, output_path: str):
    """Crops imagery based on the building polygon and the desired pixel size.
    It can zoom in to certain levels, whilst maintaining the pixel size by
    bilinear interpolation. The building will be centered in the image and all
    buildings that are cut off by the image boundary are not included.

    Parameters
    ----------
    image_array : ndarray
        The pre-event image array given from rasterio.DatasetReader.read() that underlays the building polygon.
        This image will be cropped.
    img_metadata :  dict
        The dictionary of the image metadata. See rasterio.DatasetReader.meta.
    polygon_df : object
        The building polygon that needs to be cropped around.
    zoom_level : int
        A number that dictates how large the crop_size should be, based on
        required pixel number for the model.
    pixel_num : int
        Value of the sides of the squared cropped image as input for the model.
    im_size : int
       Value of the sides of the squared input image.
    output_path : str
        Path used to save the zoomed cropped images.

    Returns
    -------
    int
        Returns whether this building polygon is red flagged and should be
        ignored, based on whether the building polygon lies on the edge of
        the image.
    """
    crop_size = int(pixel_num//zoom_level)

    polygon_grid = polygon_mask(image_array, polygon_df, im_size)
    bounding_box = mask_to_bb(polygon_grid)

    # extract bounding box information
    x1 = int(bounding_box[1])
    x2 = int(bounding_box[3])
    y1 = int(bounding_box[0])
    y2 = int(bounding_box[2])

    x_min = 0
    x_max = img_metadata["width"] - crop_size
    y_min = 0
    y_max = img_metadata["width"] - crop_size

    # UNHASH if cropped houses should be exluded
    # find out if house is on the edge and potentially cut off
    # we will not be using incomplete houses. Allow for some margin
    # as polygons will not always exactly lie on the edge
    # check for first zoom level and then break loop for that image
    # margin = 1
    # red_flag = 0
    # if (zoom_level == 1 and (x1-margin) <= x_min or (x2+margin) >= img.width
    # or (y1-margin) <= y_min or (y2+margin) >= img.width):
    #    red_flag = 1
    #    return red_flag

    # also return if house is larger than the box
    # if (y2-y1) > crop_size or (x2-x1) > crop_size:
    # find centre point of the bounding box
    x_offset = (x2+x1)//2
    y_offset = (y2+y1)//2

    # Find out dimensions of image cropping around offset
    size_limit = crop_size//2

    # find out whether the whole environment around house can be imaged
    # if not, fill up with zeroes around it
    if (y_offset + size_limit > y_max or y_offset - size_limit < y_min or
            x_offset + size_limit > x_max or x_offset - size_limit < x_min):

        # image_array = img.read()
        # padding around image, add dimensions around offset to all sides
        pad = ((0, 0), (size_limit, size_limit), (size_limit, size_limit))
        padded = np.pad(image_array, pad_width=pad)

        # transform the offset to include zero padding on all sides
        padded_x_offset = x_offset + size_limit
        padded_y_offset = y_offset + size_limit

        roi = padded[
            :,
            (padded_y_offset-size_limit):(padded_y_offset+size_limit),
            (padded_x_offset-size_limit):(padded_x_offset+size_limit)]

    else:
        # image_array = img.read()
        roi = image_array[:, (y_offset-size_limit):(y_offset+size_limit),
                          (x_offset-size_limit):(x_offset+size_limit)]

    # We want all images to be a certain pixel size, so we need to resample
    # to get zoom levels with same pixel size
    scaling_factor = pixel_num/crop_size
    # order 1 = nearest, order 2= bilinear, order 3 = cubic interpolation
    resized_img = scipy.ndimage.zoom(roi, (1, scaling_factor, scaling_factor),
                                     order=1)
    # preserve image metadata
    img_bands = img_metadata["count"]
    img_crs = img_metadata["crs"]
    img_metadata.update({
        "driver": "png",
        "height": pixel_num,
        "width": pixel_num,
        "count": img_bands,
        "crs": img_crs,
        "dtype": "uint8"})

    thread = Thread(target=save_image, args=(output_path, resized_img,
                                             img_metadata))
    thread.start()
    # with rio.open(output_path, "w", **img_metadata) as src:
    #     # Read the data from the window and write it to the output raster
    #     src.write(resized_img)


def save_image(output_path, resized_img, img_metadata):
    with rio.open(output_path, "w", **img_metadata) as src:
        # Read the data from the window and write it to the output raster
        src.write(resized_img)
    logger.debug(f"saved {output_path}")


def image_processing(zoom_levels: list, pixel_num: int):
    """Loads images and crops them based on the required zoom levels
    and the required imagery input pixel size for the model.
    Parameters
    ----------
    zoom_levels : list
        list containing all required zoom levels as integers.
    pixel_num : int
        Value of the sides of the squared cropped image as input for the model.
    """
    xbd_dir = get_xbd_dir()
    data_dir = get_data_dir()
    # where to save zoomed and cropped images
    save_dir_path = "datasets/processed_data/processed_xbd/" \
        "geotiffs_zoom/images"

    zoomdir_dict = {}
    for zoom_num in zoom_levels:
        zoom_dir = "zoom_" + str(zoom_num)
        zoom_path = os.path.join(data_dir, save_dir_path, zoom_dir)
        zoomdir_dict[zoom_num] = zoom_path
        if not os.path.exists(zoom_path):
            os.makedirs(zoom_path)

    # Load pre polygon with post damage assessment
    path_pre = os.path.join(data_dir,
                            "datasets/processed_data/metadata_pickle",
                            "xy_pre_pol_post_damage.pkl")
    if exists(path_pre) is True:
        polygons_df = pd.read_pickle(path_pre)
    else:
        # hold_filepath = get_xbd_hlabel_dir()
        hold_filepath = os.path.join(xbd_dir, "geotiffs/hold/labels")
        tier1_filepath = os.path.join(xbd_dir, "geotiffs/tier1/labels")
        tier3_filepath = os.path.join(xbd_dir, "geotiffs/tier3/labels")
        test_filepath = os.path.join(xbd_dir, "geotiffs/test/labels")
        filepaths_dict = dict.fromkeys([hold_filepath, tier1_filepath,
                                        tier3_filepath, test_filepath])
        for filepath in filepaths_dict:
            directory_files = [os.path.join(filepath, file)
                               for file in os.listdir(filepath)]
            filepaths_dict[filepath] = directory_files

        polygons_df = extract_damage_allfiles_ensemble(
            filepaths_dict=filepaths_dict,
            crs="xy")

    polygons_df["index"] = np.arange(len(polygons_df))
    filtered_df = polygons_df[["image_name", "json_link"]].drop_duplicates(
        subset=['image_name'])

    for json_path in tqdm(filtered_df["json_link"]):
        tif_name = json_path.replace("json", "tif")
        image_path = os.path.join(json_path, tif_name).replace("labels",
                                                               "images")
        # image name is in the final folder so index with -1
        name_img = json_path.rpartition("/")[-1].replace("json", "png")
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

                    # output_path = os.path.join(
                    #    zoom_dir,
                    #    (img_name.strip(".png")+"_zoom" + str(zoom_level) +
                    #     "_" + img_num))
                    output_path = os.path.join(zoom_dir, (img_num))
                    crop_images(img, polygon_for_img, zoom_level, pixel_num,
                                image_size, output_path)


def main():
    zoom_levels = [1, 2, 4, 0.5]
    pixel_num = 224
    image_processing(zoom_levels, pixel_num)


if __name__ == '__main__':
    main()