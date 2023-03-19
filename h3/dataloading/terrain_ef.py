from __future__ import annotations

import os

import affine
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import rasterio as rio
import richdem as rd
from sklearn.neighbors import BallTree
from tqdm import tqdm

from pandas.core.groupby.generic import DataFrameGroupBy

from h3 import logger
from h3.constants import EARTH_RADIUS
from h3.utils.downloader import downloader
from h3.utils.file_ops import unpack_file
from h3.utils.directories import get_processed_data_dir, get_coastline_dir, get_dem_dir, get_metadata_pickle_dir


def get_building_group() -> DataFrameGroupBy:
	"""Load the data from the building damage pickle file and group them in groups of 1 deg latlon

	Returns
	-------
	DataFrameGroupBy

	See Also
	--------
	pd.DataFrame.groupby()
	"""
	# load the data of the locations (lon, lat) of buildings
	df_pre_post_hurr_path = os.path.join(get_metadata_pickle_dir(), "lnglat_pre_pol_post_damage.pkl")
	df_pre_post_hurr = pd.read_pickle(df_pre_post_hurr_path)

	building_locs = pd.DataFrame({
		"lat": df_pre_post_hurr["geometry"].y,
		"lon": df_pre_post_hurr["geometry"].x,
		"disaster_name": df_pre_post_hurr["disaster_name"],
		"damage_class": df_pre_post_hurr["damage_class"],
	})

	# divide the building locations into groups that cover 1 degree latitude and longitude
	lon_bins = pd.cut(building_locs["lon"], bins=range(-180, 181, 1))
	lat_bins = pd.cut(building_locs["lat"], bins=range(-90, 91, 1))

	building_groups = building_locs.groupby([lon_bins, lat_bins])
	return building_groups


def _download_coastlines() -> None:
	"""Download the coastline data from Nature Earth
	(https://www.naturalearthdata.com/downloads/10m-physical-vectors/)
	"""
	# This cell check whether coastline data has been downloaded
	# You can download coastline data from Nature Earth (https://www.naturalearthdata.com/downloads/10m-physical-vectors/),
	# and store the .zip coastline data to "zip_path" (please change it according to your setting)

	# This is a weird url, but it is the correct one
	# TODO: look more into this to make sure it works
	url = ["https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip"]
	downloader(url, target_dir=get_coastline_dir())


def _unpack_coastlines(clean: bool = False) -> None:
	"""Unpack the downloaded coastline.

	Parameters
	----------
	clean : bool, optional,
		If True delete the archive file after unpack. Default is False.
	"""
	coastline_dir = get_coastline_dir()
	coastline_filename = "ne_10m_coastline.zip"
	filepath = os.path.join(coastline_dir, coastline_filename)
	unpack_file(filepath, clean=clean)


def check_coastlines_file():
	coastline_dir = get_coastline_dir()
	coastline_dir_ls = os.listdir(coastline_dir)
	if len(coastline_dir_ls) == 0:
		logger.debug("No coastline data present")
		_download_coastlines()
		_unpack_coastlines()


def _download_dem(dem_urls: list) -> None:
	"""Helper function to download the DEM to the correct folder

	Parameters
	----------
	dem_urls : list
		list of the urls to download.
	"""
	downloader(dem_urls, target_dir=get_dem_dir())


def _unpack_dem(clean: bool = False) -> None:
	"""Helper function to unpack the download DEM.zip file.

	Parameters
	----------
	clean : bool, optional
		If True, deletes the original .zip file and any .zip files extracted.
		Default is False.
	"""
	dem_dir = get_dem_dir()
	abs_list = [os.path.join(dem_dir, file) for file in os.listdir(dem_dir) if os.path.splitext(file)[1] == ".zip"]
	for file in abs_list:
		unpack_file(file, clean=clean)


def check_dem_files():
	building_group = get_building_group()
	dem_dir = get_dem_dir()
	dem_dir_ls = os.listdir(dem_dir)
	if len(dem_dir_ls) == 0:
		logger.debug("No DEM files present")
		dem_urls = get_dem_urls(building_group)
		_download_dem(dem_urls)
		_unpack_dem()


def get_coastlines() -> list[tuple[float, float]]:
	"""Load the coastline from the .shp data.
	Converts the data from the file to a list of (lat, lon)

	Returns
	-------
	list of tuple of float,
		List of tuple of the coordinates of the coastline
	"""
	shp_filename = "ne_10m_coastline.shp"
	shp_path = os.path.join(get_coastline_dir(), shp_filename)
	shapefile = gpd.read_file(shp_path)
	geometry_coast = shapefile[shapefile["geometry"].geom_type == "LineString"]["geometry"].to_numpy()
	coast_points = [(lat, lon) for point in geometry_coast for (lon, lat) in point.coords]
	return coast_points


def get_buildings_bounding_box(buildings_df: pd.DataFrame) -> tuple[int, int, int, int]:
	"""Helper function to get the bounding box of a buildings DataFrame.
	It will get the min and max of the lat lon present in the buildings_df.

	Parameters
	----------
	buildings_df : pd.DataFrame
		A pd.DataFrame of the buildings, need to have a columns labeled `lat` an `lon`.

	Returns
	-------
	tuple
		tuple of int of the coordinates of the bounding box. The values are in degrees and as follows:
		east, north, west, south
	"""
	west = int(np.floor(buildings_df["lon"].min()))
	east = int(np.ceil(buildings_df["lon"].max()))
	south = int(np.floor(buildings_df["lat"].min()))
	north = int(np.ceil(buildings_df["lat"].max()))
	return east, north, west, south


def get_coastpoints_range(bounding_box: tuple, coast_points: np.ndarray, dis_threshold: int = 2) -> np.ndarray:
	"""Function to select a subset of the coastlines points which are in range of buildings.

	Parameters
	----------
	bounding_box : tuple
		tuple of int of the coordinates of the bounding box. The values are in degrees and as follows:
		east, north, west, south. See the function get_building_bounding_box().
	coast_points : ndarray
		an array of the coast points. Coordinates are in `lat` and `lon`.
	dis_threshold : int, optional
		padding distance in degrees to the bounding box. The default is 2.

	Returns
	-------
	ndarray
		an array of the coast point dis_threshold distant to the buildings bounding box.
		Gives the coordinates in `lat`, `lon`.
	"""
	east, north, west, south = bounding_box
	mask = (
			(west - dis_threshold <= coast_points[:, 1]) &
			(coast_points[:, 1] <= east + dis_threshold) &
			(south - dis_threshold <= coast_points[:, 0]) &
			(coast_points[:, 0] <= north + dis_threshold)
	)
	return coast_points[mask]


def get_distance_coast(buildings: np.ndarray, coast_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""

	Parameters
	----------
	buildings : ndarray
		array of the buildings coordinates in `lat`, `lon`.
	coast_points : ndarray
		array of the coastline point coordinates in `lat`, `lon`.

	Returns
	-------
	nearest_coast_point : ndarray
		array of the coordinates of the coast point closest to the corresponding building point.
	dist : ndarray
		array of the distance between the nearest the coast point and corresponding building point, in metres.

	See Also
	--------
	sklearn.metrics.pairwise.haversine_distances : Compute the Haversine distance between two samples.
	sklearn.neighbors.BallTree : BallTree for nearest neighbours lookup, generalised for N-points.

	Notes
	-----
	The haversine function assumes the coordinates are latitude and longitude in radians.
	Use the following equation for the Haversine distance:

	.. math::
		D(x, y) = 2 \\arcsin{\\left(\\sqrt{\\sin^{2}{\\left(\\frac{x_1 - y_1}{2}\\right)} +
						\\cos{(x_1)}\\cos{(y_1)}\\sin^{2}{\\left(\\frac{x_2-y_2}{2}\\right)}}\\right)}

	"""
	coast_points = np.radians(coast_points)
	buildings = np.radians(buildings)
	tree = BallTree(coast_points, metric="haversine")
	dist_rad, ind = tree.query(buildings)
	dist = dist_rad * EARTH_RADIUS  # to get the distance in meters
	nearest_coast_point = np.degrees(coast_points[ind]).reshape(-1, 2)
	return nearest_coast_point, dist


def get_dem_urls(building_groups: DataFrameGroupBy) -> list:
	"""
	Generate the list of DEM to download from Land Processes Distributed Active Archive Center (LP DAAC):
	https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/.

	Parameters
	----------
	building_groups : DataFrameGroupBy
		DataFrameGroupBy object of the building

	Returns
	-------
	dem_urls : list
		list of all the urls to download.

	Notes
	-----
	The DEM files can be manually downloaded https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/.

	To automatically download them an account will be needed.
	To create the account please go to https://urs.earthdata.nasa.gov/users/new/
	"""

	dem_urls = []
	for name, group in building_groups:
		lon_floor = int(np.floor(group["lon"].min()))
		lat_floor = int(np.floor(group["lat"].min()))

		ns_str = "N" if lat_floor > 0 else "S"
		ew_str = "E" if lon_floor > 0 else "W"
		coordinate_str = f"{ns_str}{abs(lat_floor)}{ew_str}{abs(lon_floor):03}"

		dem_zip_name = f"ASTGTMV003_{coordinate_str}.zip"
		url_name = f"https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/{dem_zip_name}"
		dem_urls.append(url_name)
	return dem_urls


def lonlat2xy(lon: list, lat: list, transform: affine.Affine) -> tuple:
	# TODO: remove, to merge
	# This function convert longitude and latitude to x and y
	rows, cols = rio.transform.rowcol(transform, lon, lat)
	return cols, rows


def get_elevation(lon: list, lat: list, dem: rasterio.DatasetReader) -> np.ndarray:
	"""
	Return the elevation values for the given (lon, lat) coordinates from the provided DEM raster dataset.

	Parameters
	----------
	lon : list
		A list of longitude values of the query points.
	lat : list
		A list of latitude values of the query points.
	dem : rasterio.DatasetReader
		A rasterio dataset reader object of the DEM raster dataset.

	Returns
	-------
	elevation : np.ndarray
		A 1D numpy array of elevation values for the query points.
	"""
	coord_list = np.array((lon, lat)).T
	elevation = np.fromiter(dem.sample(coord_list, 1), dtype=np.int16)
	return elevation


def calculate_esa(building_groups: pd.DataFrameGroupBy, coast_points: np.ndarray, dem_urls: list, dis_threshold: int = 2) -> pd.DataFrame:
	esa_df = pd.DataFrame()  # to store the calculated elevation, slope and aspect
	dem_tif_path_list = [f"{os.path.basename(os.path.splitext(dem_file)[0])}_dem.tif" for dem_file in dem_urls]

	for i, (group_name, group_data) in enumerate(tqdm(building_groups)):
		tif_path = os.path.join(get_dem_dir(), dem_tif_path_list[i])
		lon, lat = group_data["lon"].values, group_data["lat"].values

		with rio.open(tif_path) as dem:
			dem_array = dem.read(1).astype("float64")
			transform = dem.transform
			elevation = get_elevation(lon, lat, dem)

		cols, rows = lonlat2xy(lon, lat, transform)

		dem4slope = rd.rdarray(dem_array, no_data=-9999)
		# defining the geotransfrom, the top left corner is (0,0), width and heigth of a pixel is 1
		dem4slope.geotransform = [0, 1, 0, 0, 0, 1, 0]
		slope_dem = rd.TerrainAttribute(dem4slope, attrib="slope_riserun", zscale=1)    # calculate slope from dem
		aspect_dem = rd.TerrainAttribute(dem4slope, attrib="aspect")                    # calculate aspect from dem

		slope = np.array(slope_dem[rows, cols])
		aspect = np.array(aspect_dem[rows, cols])

		group_esa_df = group_data.copy()
		group_esa_df["elevation"] = elevation
		group_esa_df["slope"] = slope
		group_esa_df["aspect"] = aspect

		east, north, west, south = get_buildings_bounding_box(group_data)
		points_within_range = get_coastpoints_range(
			bounding_box=(east, north, west, south),
			coast_points=coast_points,
			dis_threshold=dis_threshold
		)
		buildings = np.array([lat, lon]).T
		coast_within_range = np.column_stack((
			points_within_range[:, 0],
			points_within_range[:, 1]
		))

		nearest_coast_point, dist = get_distance_coast(buildings, coast_points=coast_within_range)
		group_esa_df["closestp_lon"] = nearest_coast_point[:, 1]
		group_esa_df["closestp_lat"] = nearest_coast_point[:, 0]
		group_esa_df["dis2coast"] = dist.flatten()

		esa_df = pd.concat([esa_df, group_esa_df], ignore_index=False)

	esa_df = esa_df.sort_index()
	return esa_df


def get_terrain_ef(esa_df: pd.DataFrame) -> None:
	terrain_efs = pd.DataFrame({
		"xbd_observation_lat": esa_df["lat"],
		"xbd_observation_lon": esa_df["lon"],
		"elevation": esa_df["elevation"],
		"slope": esa_df["slope"],
		"aspect": esa_df["aspect"],
		"dis2coast": esa_df["dis2coast"],
		"disaster_name": esa_df["disaster_name"],
		"damage_class": esa_df["damage_class"]
	})

	path_terrain_efs = os.path.join(get_processed_data_dir(), "Terrian_EFs.pkl")
	terrain_efs.to_pickle(path_terrain_efs)


def main():
	buildings_group = get_building_group()
	coast_points = np.array(get_coastlines())
	dem_urls: list = get_dem_urls(building_groups=buildings_group)
	esa_df = calculate_esa(building_groups=buildings_group, coast_points=coast_points, dem_urls=dem_urls)
	get_terrain_ef(esa_df)


if __name__ == "__main__":
	main()
