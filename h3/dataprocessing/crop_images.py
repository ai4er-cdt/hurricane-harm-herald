import numpy as np
import pandas as pd

import os
from rasterio.plot import show
import rasterio as rio
from rasterio.windows import Window
import cv2
import scipy.ndimage

from os.path import exists

from h3.utils.directories import get_xbd_dir
from h3.utils.directories import get_data_dir
from h3.dataprocessing.extract_metadata import extract_damage_allfiles_ensemble


CLASSES_DICT = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4
}


def extract_coords(row):
    return np.array(row).astype(np.int32)


def polygon_mask(img, polygon, im_size):
    img = img.read()
    image_size = (im_size, im_size)

    empty_mask = np.zeros(image_size, np.uint8)
    poly_coords = [extract_coords(polygon.exterior.coords)]
    image_mask = cv2.fillPoly(empty_mask, poly_coords, 1)
    return image_mask


def mask_to_bb(Y):
    nx, ny = np.nonzero(Y)
    # check if polygon is empty
    if len(nx) == 0 or len(ny) == 0:
        return np.zeros(4, dtype=np.float32)
    top_y = np.min(ny)
    bottom_y = np.max(ny)
    left_x = np.min(nx)
    right_x = np.max(nx)
    return np.array([left_x, top_y, right_x, bottom_y], dtype=np.int64)


def crop_images(img, polygon_df, crop_size, pixel_num, im_size, output_path):
    polygon_grid = polygon_mask(img, polygon_df, im_size)
    bounding_box = mask_to_bb(polygon_grid)

    # extract bounding box information
    x1 = int(bounding_box[1])
    x2 = int(bounding_box[3])
    y1 = int(bounding_box[0])
    y2 = int(bounding_box[2])

    x_min = 0
    x_max = img.width - crop_size
    y_min = 0
    y_max = img.width - crop_size

    # find out if house is on the edge and potentially cut off
    # we will not be using incomplete houses. Allow for some margin
    # as polygons will not always exactly lie on the edge

    margin = 1
    if ((x1-margin) <= x_min or (x2+margin) >= img.width or
            (y1-margin) <= y_min or (y2+margin) >= img.width):
        return
    # also return if house is larger than the box
    if (y2-y1) > crop_size or (x2-x1) > crop_size:
        return
    else:
        # find centre point of the bounding box
        x_offset = (x2+x1)//2
        y_offset = (y2+y1)//2

    # Find out dimensions of image cropping around offset
    size_limit = crop_size//2

    # find out whether the whole environment around house can be imaged
    # if not, fill up with zeroes around it
    if (y_offset + size_limit > y_max or y_offset - size_limit < y_min or
            x_offset + size_limit > x_max or x_offset - size_limit < x_min):

        image_array = img.read()

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
        image_array = img.read()
        roi = image_array[:, (y_offset-size_limit):(y_offset+size_limit),
                          (x_offset-size_limit):(x_offset+size_limit)]

    # We want all images to be a certain pixel size, so we need to resample
    # to get zoom levels with same pixel size

    scaling_factor = pixel_num/crop_size
    # order 1 = nearest, order 2= bilinear, order 3 = cubic interpolation
    resized_img = scipy.ndimage.zoom(roi, (1, scaling_factor, scaling_factor),
                                     order=1)

    # preserve image metadata
    img_bands = img.meta["count"]
    img_crs = img.meta["crs"]
    metadata_profile = img.profile
    metadata_profile.update({
        "height": pixel_num,
        "width": pixel_num,
        "count": img_bands,
        "crs": img_crs})

    with rio.open(output_path, "w", **metadata_profile) as src:
        # Read the data from the window and write it to the output raster
        src.write(resized_img)
        print(output_path)


def image_processing(zoom_levels, pixel_num):
    xbd_dir = get_xbd_dir()
    data_dir = get_data_dir()
    labels_path = "geotiffs/hold/labels/"

    filepath = os.path.join(xbd_dir, labels_path, "")

    # where to save zoomed and cropped images
    save_dir_path = "datasets/processed_data/geotiffs_zoom/hold/images"
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
        fulldirectory_files = [os.path.join(filepath, file)
                               for file in os.listdir(filepath)]
        polygons_df = extract_damage_allfiles_ensemble(fulldirectory_files,
                                                       filepath,
                                                       "lng_lat")
    for building_num in range(len(polygons_df[:5])):
        building_geometry = (polygons_df.iloc[building_num]["geometry"])
        # find image name
        img_name = (polygons_df.iloc[building_num]["image_name"])
        tif_name = img_name.replace("png", "tif")
        image_path = os.path.join(filepath, tif_name).replace("labels",
                                                              "images")

        # load building image
        with rio.open(image_path) as img:
            image_size = 1024
            for zoom_level in zoom_levels:
                zoom_dir = zoomdir_dict[zoom_level]
                crop_size = pixel_num//zoom_level
                img_num = str(building_num) + ".png"
                output_path = os.path.join(zoom_dir, img_num)
                crop_images(img, building_geometry, crop_size, pixel_num,
                            image_size, output_path)


def main():
    zoom_levels = [1, 2, 4]
    pixel_num = 224
    image_processing(zoom_levels, pixel_num)


if __name__ == '__main__':
    main()
