from __future__ import annotations

import os
import numpy as np
import rasterio

from typing import Container, Sequence, Iterable, Literal

from h3 import logger
from h3.utils.downloader import downloader
from h3.utils.directories import get_storm_dir
from h3.utils.file_ops import unpack_file
from h3.constants import HAITI_LAT_MAX, HAITI_LAT_MIN, HAITI_LON_MAX, HAITI_LON_MIN
from h3.constants import TEXAS_TO_MAINE_LAT_MAX, TEXAS_TO_MAINE_LAT_MIN, TEXAS_TO_MAINE_LON_MAX, TEXAS_TO_MAINE_LON_MIN


def _download_storm() -> None:
	"""Private wrapper function around downloader() from h3.utils.downloader for the storm surge files
	downloading the files in the `storm_dir`.

	See Also
	--------
	h3.utils.downloader.downloader()
	"""
	url = [
		"https://www.nhc.noaa.gov/gis/hazardmaps/US_SLOSH_MOM_Inundation_v3.zip",
		"https://www.nhc.noaa.gov/gis/hazardmaps/Hispaniola_SLOSH_MOM_Inundation.zip"
	]
	downloader(url, target_dir=get_storm_dir())


def _unpack_storm(clean: bool = False) -> None:
	storm_dir = get_storm_dir()
	abs_list = [os.path.join(storm_dir, file) for file in os.listdir(storm_dir) if os.path.splitext(file)[1] == ".zip"]
	for p in abs_list:
		unpack_file(p, clean=clean)


def check_storm(clean_after_unpack: bool = False, reload: bool = False) -> None:
	"""Helper function to check if files are present in storm_dir
	It is quite naive function, as it will not check what files are present, but only if any files are present.
	If no files are present, it will download and unpack them.

	Parameters
	----------
	clean_after_unpack : bool, optional
		If True, will delete the zip downloaded files after unpacking them. The default is False.
	reload : bool, optional
		If True, force the re-download and unpack,
	"""
	storm_dir = get_storm_dir()
	all_files = os.listdir(storm_dir)
	# TODO: reload would delete existing file to prevent unexpected behaviour with overwriting
	if len(all_files) == 0 or reload:
		logger.debug("Storm surge files are not present.\nDownloading them.")
		_download_storm()
		# TODO: check this with the unpacking helper function
		logger.debug("unpacking the files")
		_unpack_storm(clean=clean_after_unpack)


def get_location_box(location: Literal["us", "haiti"]) -> None | tuple:
	"""Get the location box.

	Parameters
	----------
	location : str, {"us", "haiti"}
		location to get the box coordinate in lat lon


	Returns
	-------
	tuple, optional
		a tuple of the coordinates of the location, as follows:
		LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
	"""
	if location == "us":
		return TEXAS_TO_MAINE_LON_MIN, TEXAS_TO_MAINE_LON_MAX, TEXAS_TO_MAINE_LAT_MIN, TEXAS_TO_MAINE_LAT_MAX
	elif location == "haiti":
		return HAITI_LON_MIN, HAITI_LON_MAX, HAITI_LAT_MIN, HAITI_LAT_MAX


def is_in_area(point: Sequence[float], area_box: Container[float]) -> bool:
	"""Check if point is in an area bounding box.

	Parameters
	----------
	point : tuple_like
		A tuple_like of the lat, lon of the point to check.
	area_box : tuple_like
		A tuple_like of the bounding box of the area to check.
		It needs to be as follows:
		LON_MIN, LON_MAX, LAT_MIN, LAT_MAX
		Usually this bounding box is given by the get_location_box function.

	Returns
	-------
	bool
		True if the point is in the bounding box
	"""
	# TODO: check the types
	lon_min, lon_max, lat_min, lat_max = area_box
	return (lat_min <= point[0] <= lat_max) and (lon_min <= point[1] <= lon_max)


def point_locations(lat: float, lon: float, locations: Iterable[str]) -> None | str:
	"""Get the location name of a point given its latitude and longitude

	Parameters
	----------
	lat : float
		latitude of the point to check
	lon : float
		longitude of the point to check
	locations : Iterable of str
		Iterable-like of the name of the different locations to check.

	Returns
	-------
	str, optional
		Name of the location areas of the point.

	Examples
	--------
	>>> lat, lon = 18, -72  # this point is in Haiti
	>>> point_locations(lat, lon, ["us", "haiti"])
	haiti
	"""
	point = (lat, lon)
	location: Literal["us", "haiti"]    # Prevent PyCharm to be annoying with the Literal type
	for location in locations:
		location_box = get_location_box(location)
		if is_in_area(point, location_box):
			return location


def latlon2_storm_surge(coords: Iterable[Iterable[float, float]], category: int) -> list[int]:
	storm_dir = get_storm_dir()
	us_filename = f"us_Category{category}_MOM_Inundation_HIGH.tif"
	haiti_filename = f"Hispaniola_Category{category}_MOM_Inundation_HighTide.tif"

	us_filepath = os.path.join(storm_dir, "US_SLOSH_MOM_Inundation_v3", us_filename)
	haiti_filepath = os.path.join(storm_dir, "Hispaniola_SLOSH_MOM_Inundation_11_2018", haiti_filename)

	us_image = rasterio.open(us_filepath)
	haiti_image = rasterio.open(haiti_filepath)

	locations = ["us", "haiti"]
	image_transform = {
		"us": us_image.transform,
		"haiti": haiti_image.transform,
	}

	# get the location of the point
	storm_surge = []
	for lat, lon in coords:
		location = point_locations(lat, lon, locations)
		if location is not None:
			row, col = rasterio.transform.rowcol(image_transform[location], lon, lat)
			window = rasterio.windows.Window(col, row, 1, 1)

		if location == "us":
			surge_value = us_image.read(1, window=window).item()
		elif location == "haiti":
			surge_value = haiti_image.read(1, window=window).item()
		else:
			surge_value = np.NaN
		storm_surge.append(surge_value)

	return storm_surge


def main():
	coords = [
		[18.638641, -72.331067],
		[18.553196, -72.408681],
		[18.553752, -72.380271],
		[28.157927, -80.587846],
		[32.214255, -80.675551],
		[32.185472, -80.606953],
		[67.185472, -90.606953]
	]
	for cat in range(1, 6):
		storm_values = latlon2_storm_surge(coords, category=cat)
		print(cat, *storm_values)


if __name__ == "__main__":
	main()
