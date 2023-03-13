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

from pandas.core.groupby.generic import DataFrameGroupBy

from h3.constants import EARTH_RADIUS
from h3.utils.downloader import downloader
from h3.utils.file_ops import unpack_file
from h3.utils.directories import get_processed_data_dir, get_coastline_dir, get_dem_dir, get_metadata_pickle_dir



def get_building_group():
	# load the data of the locations (lon, lat) of buildings
	df_pre_post_hurr_path = os.path.join(get_metadata_pickle_dir(), "lnglat_pre_pol_post_damage.pkl")
	# building_locs_path="/content/drive/MyDrive/ai4er/python/hurricane/hurricane-harm-herald/data/datasets/xBD_data/xbd_points_posthurr_reformatted.pkl"
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
	# This cell check whether coastline data has been downloaded
	# You can download coastline data from Nature Earth (https://www.naturalearthdata.com/downloads/10m-physical-vectors/),
	# and store the .zip coastline data to "zip_path" (please change it according to your setting)

	# zip_path = "/data/datasets/EFs/terrain_data/ne_10m_coastline.zip"
	# shp_path = "/data/datasets/EFs/terrain_data/ne_10m_coastline/ne_10m_coastline.shp"
	# coast_extracted_floder = "/data/datasets/EFs/terrain_data/ne_10m_coastline/"
	# pkl_path = "/data/datasets/EFs/terrain_data/ne_10m_coastline/ne_10m_coastline.pkl"

	# This is a weird url, but it is the correct one
	# TODO: look more into this to make sure it works
	url = ["https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip"]
	downloader(url, target_dir=get_terrain_dir())


def _unpack_coastlines(clean: bool = False):
	coastline_dir = get_coastline_dir()
	filepath = os.path.join(coastline_dir, "ne_10m_coastline.zip")
	unpack_file(filepath, clean=clean)

# check whether the .shp or .zip coastline data exist
# if os.path.isfile(shp_path):
# 	print(".shp coastline file found")
# else:
# 	if os.path.isfile(zip_path):
# 		with zipfile.ZipFile(zip_path, "r") as zip_ref:
# 			zip_ref.extractall(coast_extracted_floder)  # the coastline data is in 10m resolution
# 	else:
# 		print(".zip coastline file not found, please download it from ", url, " manually")


def get_coastlines():
	# Load the .shp coastline data
	# Convert the .shp coast line data into points and store in a dataframe
	shp_path = os.path.join(get_terrain_dir(), "ne_10m_coastline.shp")
	shapefile = gpd.read_file(shp_path)                                 # Read the shapefile

	geometry_coast = shapefile[shapefile["geometry"].geom_type == "LineString"]["geometry"].to_numpy()
	coast_points = [(lon, lat) for point in geometry_coast for (lon, lat) in point.coords]
	return coast_points


# Find the closest point on the coastline to the building and return the distance
# NOTE! this cell is quite time consuming to run (dependding on the size of building location data and the coastline data)

# define the function that calculate Geodesic distance between two points
def geoddist(p1, p2):
	return Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])["s12"]


def another_plot(building_groups, coast_points, dis_threshold: int = 2, calculate_dis_to_coast: bool = True):
	# assuming the building is not more than dis_threshold latitude and longitude away from the coast,

	coast_points = np.array(coast_points)
	n_groups = len(building_groups)  #
	n_cols = 3
	n_rows = math.ceil(n_groups / n_cols)
	fig, axs = plt.subplots(
		n_rows,
		n_cols,
		figsize=(12, 12),
		dpi=300,
		subplot_kw={"projection": ccrs.PlateCarree()}
	)

	for i, (group_name, group_data) in enumerate(building_groups):
		west = int(np.floor(group_data["lon"].min()))
		east = int(np.ceil(group_data["lon"].max()))
		south = int(np.floor(group_data["lat"].min()))
		north = int(np.ceil(group_data["lat"].max()))

		# chop the coastline data
		mask = (
			(west - dis_threshold <= coast_points[:, 0]) &
			(coast_points[:, 0] <= east + dis_threshold) &
			(south - dis_threshold <= coast_points[:, 1]) &
			(coast_points[:, 1] <= north + dis_threshold)
		)
		points_within_range = coast_points[mask]

		# plot the buildings and the coastline data that been choped
		row = i // n_cols
		col = i % n_cols
		ax = axs[row, col]
		ax.set_xlim(west - dis_threshold, east + dis_threshold)
		ax.set_ylim(south - dis_threshold, north + dis_threshold)
		ax.add_feature(cfeature.LAND.with_scale("10m"))
		ax.add_feature(cfeature.OCEAN.with_scale("10m"))

		# Plot the coast points
		ax.scatter(points_within_range[:, 0], points_within_range[:, 1], s=5, transform=ccrs.PlateCarree())
		ax.scatter(group_data["lon"], group_data["lat"], s=5, transform=ccrs.PlateCarree())

		# Set x-label and y-label
		ax.set_xlabel("Longitude (°)", fontsize=12)
		ax.set_ylabel("Latitude (°)", fontsize=12)

		# Set x-ticks and y-ticks
		xticks = np.arange(west - dis_threshold, east + dis_threshold, dis_threshold)
		yticks = np.arange(south - dis_threshold, north + dis_threshold, dis_threshold)

		ax.set_xticks(xticks, crs=ccrs.PlateCarree())
		ax.set_yticks(yticks, crs=ccrs.PlateCarree())

		points_within_range = np.column_stack((points_within_range[:, 0], points_within_range[:, 1]))
		buildings = np.column_stack((group_data["lon"], group_data["lat"]))

		if calculate_dis_to_coast:
			closestp = np.zeros([len(buildings), 2])  # to store the closest point in the coast to a given building
			distance = np.zeros([len(buildings), 1])  # to store the building's distance to the coast
			coast_vp = vptree.VPTree(points_within_range, geoddist)  # build the lookup table

			for j in range(len(buildings)):
				data = coast_vp.get_nearest_neighbor(
					buildings[j, :])  # find buildings' closest point on the coastline and get the distance
				closestp[j, :] = data[1]
				distance[j] = data[0]  # distance in unit of meter

			building_locs.loc[group_data.index, "closestp_lon"] = closestp[:, 0]  # store the calculated data into "building_locs" dataframe
			building_locs.loc[group_data.index, "closestp_lat"] = closestp[:, 1]
			building_locs.loc[group_data.index, "dis2coast"] = distance[:, 0]

			ax.scatter(closestp[:, 0], closestp[:, 1], s=5, transform=ccrs.PlateCarree(), c="yellow")
			ax.plot([buildings[:, 0], closestp[:, 0]], [buildings[:, 1], closestp[:, 1]], "k", linewidth=0.1)

	for i in range(n_cols * n_rows):
		fig.delaxes(axs.flatten()[i + 1])
	plt.show()

# check the data
building_locs

def get_dem_urls(building_groups):
	# https://urs.earthdata.nasa.gov/users/new/
	# The following code check whether DEM data has been downloaded
	# If it is not, the DEM files can be downloaded from Land Processes Distributed Active Archive Center (LP DAAC) manually
	# The link to download DEM files: https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/
	# If you have not download, please run this cell first to print the list of DEM files to download.
	# Then, put DEM files in a local floder and update the "dem_zip_path"
	# and run the code, DEM files will ben extracted to "extracted_path" (please change it according to your setting)
	dem_tif_name_list = []
	dem_tif_path_list = []
	dem_tif_short_name_list = []

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

		# PLEASE MAUALLY DOWNLOAD THE DEM FILE SPECIFIED BY "dem_zip_name" AND PUT IT INTO THE "DEM_ZIP_PATH"
		# dem_zip_path = f"./data/datasets/EFs/terrain_data/DEM_data/{dem_zip_name}"  # path to store the downloaded data, please change it accordingly
		# dem_tif_name = f"ASTGTMV003_{coordinate_str}_dem.tif"
		# dem_tif_short_name = f"{coordinate_str}"
		# # extracted_path = "./data/datasets/EFs/terrain_data/DEM_data/DEM_extracted"  # path to store the extracted data, please change it accordingly
		# # dem_tif_path = f"{extracted_path}/{dem_tif_name}"
		#
		# dem_tif_name_list.append(dem_tif_name)
		# dem_tif_short_name_list.append(dem_tif_short_name)
		# dem_tif_path_list.append(dem_tif_path)

		# Check if the .tif file already exists in the specified directory
		# if os.path.isfile(dem_tif_path):
		# 	continue
		# else:
		# 	if os.path.isfile(dem_zip_path):
		# 		with zipfile.ZipFile(dem_zip_path, "r") as zip_ref:
		# 			zip_ref.extract(dem_tif_name, extracted_path)
		# 	else:
		# 		print("DEM file Not found, please download: ", dem_zip_name)

	return dem_urls


def _download_dem(dem_urls) -> None:
	downloader(dem_urls, target_dir=get_dem_dir())


def _unpack_dem(clean: bool = False) -> None:
	dem_dir = get_dem_dir()
	abs_list = [os.path.join(dem_dir, file) for file in os.listdir(dem_dir) if os.path.splitext(file)[1] == ".zip"]
	for file in abs_list:
		unpack_file(file, clean=clean)


def lonlat2xy(lon: list, lat: list, transform: affine.Affine) -> tuple:
	# TODO: remove, to merge
	# This function convert longitude and latitude to x and y
	rows, cols = rio.transform.rowcol(transform, lon, lat)
	return cols, rows


def get_elevation(lon: list, lat: list, dem: rasterio.DatasetReader) -> np.ndarray:  # get the height of the given location (given by lon and lat)
	coord_list = np.array((lon, lat)).T
	elevation = np.fromiter(dem.sample(coord_list, 1), dtype=np.int16)
	# df["elevation"] = np.array(data)
	return elevation


# The following code call corresponding functions and calculate the elevation, slope and aspect for buildings

def calculate_esa(building_groups, dem_urls):
	esa_df = pd.DataFrame()  # to store the calculated elevation, slope and aspect
	dem_tif_path_list = [os.path.basename(os.path.splitext(dem_file)[0]) for dem_file in dem_urls]

	for i, (group_name, group_data) in enumerate(building_groups):
		tif_path = os.path.join(get_dem_dir(), f"{dem_tif_path_list[i]}_dem.tif")
		lon, lat = group_data["lon"].values, group_data["lat"].values

		with rio.open(tif_path) as dem:
			dem_array = dem.read(1).astype("float64")
			transform = dem.transform
			elevation = get_elevation(lon, lat, dem)

		cols, rows = lonlat2xy(lon, lat, transform)

		dem4slope = rd.rdarray(dem_array, no_data=-9999)
		dem4slope.geotransform = [0, 1, 0, 0, 0, 1, 0]  # defining the geotransfrom, the top left corner is (0,0), width and heigth of a pixel is 1
		slope_dem = rd.TerrainAttribute(dem4slope, attrib="slope_riserun", zscale=1)  # calculate slope
		aspect_dem = rd.TerrainAttribute(dem4slope, attrib="aspect")  # calculate aspect

		# temp_df = group_data
		# temp_df = get_height(group_data, dem)  # calculate height
		#
		# slope = get_slope(cols, rows, slope_dem)  # calculate slope
		# aspect = get_aspect(cols, rows, aspect_dem)  # calculate aspect
		slope = np.array(slope_dem[rows, cols])
		aspect = np.array(aspect_dem[rows, cols])

		esa_df = pd.concat([esa_df, group_data], ignore_index=False)

	esa_df = esa_df.sort_index()

	return esa_df


def get_terrain_ef(esa_df):
	# Explore the data to a local path

	Terrian_EFs = pd.DataFrame({
		"xbd_observation_lat": esa_df["lat"],
		"xbd_observation_lon": esa_df["lon"],
		"elevation": esa_df["elevation"],
		"slope": esa_df["slope"],
		"aspect": esa_df["aspect"],
		"dis2coast": esa_df["dis2coast"],
		"disaster_name": esa_df["disaster_name"],
		"damage_class": esa_df["damage_class"]
	})

	# path to store Terrain_EFs
	path_Terrain_EFs = os.path.join(get_processed_data_dir(), "Terrian_EFs.pkl")
	Terrian_EFs.to_pickle(path_Terrain_EFs)  # store the dataframe


	# read the stored data for test
	with open(path_Terrain_EFs, "rb") as f:
		Terrian_EFs_test = pickle.load(f)



def main():
	pass


if __name__ == "__main__":
	main()
