from __future__ import annotations

import os
import math
import pickle
import zipfile

import affine
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio as rio
from rasterio.plot import show

import richdem as rd

from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

import vptree
from geographiclib.geodesic import Geodesic

from affine import Affine
import shapefile
import cartopy.feature as cfeature

from h3.utils.downloader import downloader
from h3.utils.file_ops import unpack_file
from h3.utils.directories import get_terrain_dir, get_processed_data_dir, get_coastline_dir, get_dem_dir, get_metadata_pickle_dir


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


def building_plot(building_groups):
	# plot the building locations for verification
	n_groups = len(building_groups)  # group number
	n_cols = 3  # column number
	n_rows = math.ceil(n_groups / n_cols)  # raw number

	fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12), dpi=300, subplot_kw={"projection": ccrs.PlateCarree()})

	for i, (group_name, group_data) in enumerate(building_groups):
		west = int(np.floor(group_data["lon"].min()))
		east = int(np.ceil(group_data["lon"].max()))
		south = int(np.floor(group_data["lat"].min()))
		north = int(np.ceil(group_data["lat"].max()))
		dis_threshold = 1

		# plot the buildings and the coastline data that been choped
		row = i // n_cols
		col = i % n_cols
		ax = axs[row, col]
		ax.set_xlim(west - dis_threshold, east + dis_threshold)
		ax.set_ylim(south - dis_threshold, north + dis_threshold)

		ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5)
		ax.add_feature(cfeature.LAND.with_scale("10m"))
		ax.add_feature(cfeature.OCEAN.with_scale("10m"))

		# plot the locations of buildings
		ax.scatter(group_data["lon"], group_data["lat"], s=5, transform=ccrs.PlateCarree(), c="orange")

		# Set x-label and y-label
		ax.set_xlabel("Longitude (°)", fontsize=12)
		ax.set_ylabel("Latitude (°)", fontsize=12)

		# Set x-ticks and y-ticks
		xticks = np.arange(west - dis_threshold, east + dis_threshold, dis_threshold)
		yticks = np.arange(south - dis_threshold, north + dis_threshold, dis_threshold)
		ax.set_xticks(xticks, crs=ccrs.PlateCarree())
		ax.set_yticks(yticks, crs=ccrs.PlateCarree())

	# while i < n_cols * n_rows - 1:
	for i in range(n_cols * n_rows):
		fig.delaxes(axs.flatten()[i + 1])

	plt.show()


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
	# 234 ms ± 4.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

	# coast_points = pd.DataFrame(columns=["coast_lon", "coast_lat"])     # Initialize the coast_points DataFrame
	# temp_array = []      # Initialize the temp_array as  a list
	#
	# for i, row in shapefile.iterrows():
	# 	if row["geometry"].geom_type == "LineString":
	# 		for point in row["geometry"].coords:
	# 			x, y = point
	# 			p = [x, y]                                                          # Create a list with two elements
	# 			temp_array.append(p)                                                # Append the point to the list
	# 		temp_df = pd.DataFrame(temp_array, columns=["coast_lon", "coast_lat"])  # Create a DataFrame from the temp_array
	# 		coast_points = pd.concat([coast_points, temp_df],
	# 		                         ignore_index=True)  # Append the temp_df to the coast_points DataFrame
	# 		temp_array = []  # Reset the temp_array to an empty list
	# 9.39 s ± 31.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


def plot_coastline(coast_points):
	# Plot the coastline data for verification
	fig = plt.figure(figsize=(12, 6), dpi=300)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	# Add a global map background
	ax.stock_img()
	# Plot the coast points
	ax.scatter(coast_points["coast_lon"], coast_points["coast_lat"], s=5, transform=ccrs.PlateCarree())
	# Set x-label and y-label
	ax.set_xlabel("Longitude (°)", fontsize=12)
	ax.set_ylabel("Latitude (°)", fontsize=12)
	# Set x-ticks and y-ticks
	xticks = np.arange(-180, 190, 20)
	yticks = np.arange(-90, 100, 20)
	ax.set_xticks(xticks, crs=ccrs.PlateCarree())
	ax.set_yticks(yticks, crs=ccrs.PlateCarree())

	plt.show()

###

# Find the closest point on the coastline to the building and return the distance
# NOTE! this cell is quite time consuming to run (dependding on the size of building location data and the coastline data)

# define the function that calculate Geodesic distance between two points
def geoddist(p1, p2):
	return Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])["s12"]


def another_plot():
	n_groups = len(building_groups)  #
	n_cols = 3
	n_rows = math.ceil(n_groups / n_cols)
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12), dpi=300,
	                        subplot_kw={"projection": ccrs.PlateCarree()})

	for i, (group_name, group_data) in enumerate(building_groups):
		west = int(np.floor(group_data["lon"].min()))
		east = int(np.ceil(group_data["lon"].max()))
		south = int(np.floor(group_data["lat"].min()))
		north = int(np.ceil(group_data["lat"].max()))

	dis_threshold = 2  # assuming the building is not more than dis_threshold latitude and longitude away from the coast, please change it according to your case

		# chop the coastline data
		mask = (coast_points["coast_lon"] >= west - dis_threshold) & (coast_points["coast_lon"] <= east + dis_threshold) & \
		       (coast_points["coast_lat"] >= south - dis_threshold) & (coast_points["coast_lat"] <= north + dis_threshold)
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
		ax.scatter(points_within_range["coast_lon"], points_within_range["coast_lat"], s=5, transform=ccrs.PlateCarree())
		ax.scatter(group_data["lon"], group_data["lat"], s=5, transform=ccrs.PlateCarree())
		# Set x-label and y-label
		ax.set_xlabel("Longitude (°)", fontsize=12)
		ax.set_ylabel("Latitude (°)", fontsize=12)
		# Set x-ticks and y-ticks
		xticks = np.arange(west - dis_threshold, east + dis_threshold, dis_threshold)
		yticks = np.arange(south - dis_threshold, north + dis_threshold, dis_threshold)

		ax.set_xticks(xticks, crs=ccrs.PlateCarree())
		ax.set_yticks(yticks, crs=ccrs.PlateCarree())

		points_within_range = np.column_stack((points_within_range["coast_lon"], points_within_range["coast_lat"]))
		buildings = np.column_stack((group_data["lon"], group_data["lat"]))

		calculate_dis_to_coast = True  # the switch for whether or not to calculate the distance to the coast
		if calculate_dis_to_coast:
			closestp = np.zeros([len(buildings), 2])  # to store the closest point in the coast to a given building
			distance = np.zeros([len(buildings), 1])  # to store the building"s distance to the coast
			coast_vp = vptree.VPTree(points_within_range, geoddist)  # build the lookup table

			for i in range(0, len(buildings)):
				data = coast_vp.get_nearest_neighbor(
					buildings[i, :])  # find buildings" closest point on the coast line and get the distance
				closestp[i, :] = data[1]
				distance[i] = data[0]  # distance in unit of meter

			building_locs.loc[group_data.index, "closestp_lon"] = closestp[:,
			                                                      0]  # store the calculated data into "building_locs" dataframe
			building_locs.loc[group_data.index, "closestp_lat"] = closestp[:, 1]
			building_locs.loc[group_data.index, "dis2coast"] = distance[:, 0]

			ax.scatter(closestp[:, 0], closestp[:, 1], s=5, transform=ccrs.PlateCarree(), c="yellow")
			ax.plot([buildings[:, 0], closestp[:, 0]], [buildings[:, 1], closestp[:, 1]], "k", linewidth=0.1)
	while i < n_cols * n_rows - 1:
		fig.delaxes(axs.flatten()[i + 1])
		i += 1
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


# This cell plot DEM files
def plot_dem():
	# Set the number of columns and rows for the plot
	num_cols = 3
	num_rows = -(-len(dem_tif_path_list) // num_cols)

	# Create a new figure with the appropriate number of subplots
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=300)

	# Iterate over the files and plot each one in a subplot
	for i, file in enumerate(dem_tif_path_list):
		row, col = divmod(i, num_cols)
		ax = axs[row, col]
		with rio.open(file) as dem:
			dem_array = dem.read(1).astype("float64")
			handle = rio.plot.show(
				dem_array,
				transform=dem.transform,
				ax=ax,
				title=f"{dem_tif_short_name_list[i]}",
				cmap="gist_earth",
				vmin=0,
				vmax=np.percentile(dem_array, 99)
			)  # plot DEM map

			im = handle.get_images()[0]
			cbar = fig.colorbar(im, ax=ax)
			cbar.set_label("Elevation (m)")
			ax.set_xlabel("Longitude(°)")
			ax.set_ylabel("Latitude(°)")

	# Remove any unused subplots
	for i in range(len(axs.flat)):
		if i >= len(dem_tif_path_list):
			fig.delaxes(axs.flat[i])

	plt.show()


# This cell define functions to get the height, slope and aspect for buildings

def lonlat2xy(lon, lat, dem):  # This function convert longitude and latitude to x and y
	rows, cols = rio.transform.rowcol(dem.transform, lon, lat)
	return cols, rows


def get_height(df, dem):  # get the height of the given location (given by lon and lat)
	coord_list = [(x, y) for x, y in zip(df["lon"], df["lat"])]
	data = [x for x in dem.sample(coord_list, 1)]
	df["elevation"] = np.array(data)
	return df


# def get_height_v2(cols,rows,dem_array): # get the height of the given location (given by lon and lat)
#   x=np.arange(0,np.size(dem_array,1),1)
#   y=np.arange(0,np.size(dem_array,0),1)
#   f = interp2d(x,y,dem_array, kind="linear") #interploate
#   H_all=np.zeros([len(cols),1])
#   for i in range(0,len(cols)):  # do the interplotation one by one, otherwise the interp2d will sort the data
#      height=f(cols[i],rows[i])
#      H_all[i]=height
#   return H_all

def get_slope_aspect(cols, rows, terrain_attribute):  # get the slope of the given location (given by lon and lat)
	x = np.arange(0, np.size(terrain_attribute, 1), 1)
	y = np.arange(0, np.size(terrain_attribute, 0), 1)
	interp = RegularGridInterpolator(
		points=(x, y),
		values=np.array(terrain_attribute).astype("float64"),
		method="linear",
		bounds_error=False,
		fill_value=None
	)
	s_all = interp((rows, cols))
	# f = interp2d(x, y, slope, kind="linear",)  # interploate
	# s_all = [f(c, r) for c, r in zip(cols, rows)]
	#
	# S_all = np.zeros([len(cols), 1])
	# for i in range(0, len(cols)):  # do the interplotation one by one, otherwise the interp2d will sort the data
	# 	s = f(cols[i], rows[i])
	# 	S_all[i] = s
	return s_all


# def get_aspect(cols, rows, aspect):  # get the aspect of the given location (given by lon and lat)
# 	x = np.arange(0, np.size(aspect, 1), 1)
# 	y = np.arange(0, np.size(aspect, 0), 1)
# 	f = interp2d(x, y, aspect, kind="linear")  # interploate
# 	# f = RegularGridInterpolator((x,y),aspect) #interploate
# 	a_all = [f(c, r) for c, r in zip(cols, rows)]
# 	A_all = np.zeros([len(cols), 1])
# 	for i in range(0, len(cols)):  # do the interplotation one by one, otherwise the interp2d will sort the data
# 		a = f(cols[i], rows[i])
# 		A_all[i] = a
# 	return a_all


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
		slope = get_slope_aspect(cols, rows, terrain_attribute=slope_dem)  # calculate slope
		aspect = get_slope_aspect(cols, rows, terrain_attribute=aspect_dem)  # calculate aspect

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


def other_plot():
	# plot dem map and building locations
	# Set the number of columns and rows for the plot
	num_cols = 3
	num_rows = -(-len(dem_tif_path_list) // num_cols)

	# Create a new figure with the appropriate number of subplots
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=300)

	for i, (group_name, group_data) in enumerate(building_groups):
		with rio.open(dem_tif_path_list[i]) as dem:
			dem_array = dem.read(1).astype("float64")
			row, col = divmod(i, num_cols)
			ax = axs[row, col]
			handle = rio.plot.show(dem_array, transform=dem.transform, ax=ax, title=f"{dem_tif_short_name_list[i]}",
			                       cmap="gist_earth", vmin=0, vmax=np.percentile(dem_array, 99))  # plot DEM map
			temp_df = group_data
			gdf = gpd.GeoDataFrame(temp_df, geometry=gpd.points_from_xy(temp_df.lon, temp_df.lat), crs=dem.crs)
			gdf.plot(ax=handle, color="red")  # plot location of buildings
			im = handle.get_images()[0]
			cbar = fig.colorbar(im, ax=ax)
			cbar.set_label("Elevation (m)")
			ax.set_xlabel("Longitude(°)")
			ax.set_ylabel("Latitude(°)")

	# Remove any unused subplots
	for i in range(len(axs.flat)):
		if i >= len(dem_tif_path_list):
			fig.delaxes(axs.flat[i])

	plt.show()


def plot():
	# plot the slope and building locations
	# Set the number of columns and rows for the plot
	num_cols = 3
	num_rows = -(-len(dem_tif_path_list) // num_cols)

	# Create a new figure with the appropriate number of subplots
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=200)

	for i, (group_name, group_data) in enumerate(building_groups):
		with rio.open(dem_tif_path_list[i]) as dem:
			dem_array = dem.read(1).astype("float64")
			row, col = divmod(i, num_cols)
			ax = axs[row, col]
			dem4slope = rd.rdarray(dem_array, no_data=-9999)
			dem4slope.geotransform = [0, 1, 0, 0, 0, 1,
			                          0]  # defining the geotransfrom, the top left corner is (0,0), width and heigth of a pixel is 1
			slope = rd.TerrainAttribute(dem4slope, attrib="slope_riserun")  # calculate slope
			handle = rio.plot.show(slope, transform=dem.transform, ax=ax, title=f"{dem_tif_short_name_list[i]}",
			                       cmap="PuBu", vmin=0, vmax=np.percentile(slope, 95))  # plot DEM map
			temp_df = group_data
			gdf = gpd.GeoDataFrame(temp_df, geometry=gpd.points_from_xy(temp_df.lon, temp_df.lat), crs=dem.crs)
			gdf.plot(ax=handle, color="red")  # plot location of buildings
			im = handle.get_images()[0]
			cbar = fig.colorbar(im, ax=ax)
			cbar.set_label("Slope (%)")
			ax.set_xlabel("Longitude(°)")
			ax.set_ylabel("Latitude(°)")

	# Remove any unused subplots
	for i in range(len(axs.flat)):
		if i >= len(dem_tif_path_list):
			fig.delaxes(axs.flat[i])

	# Show the figure
	plt.show()


def p():
	# plot the aspect and building locations
	# Set the number of columns and rows for the plot
	num_cols = 3
	num_rows = -(-len(dem_tif_path_list) // num_cols)

	# Create a new figure with the appropriate number of subplots
	fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=200)

	for i, (group_name, group_data) in enumerate(building_groups):
		with rio.open(dem_tif_path_list[i]) as dem:
			dem_array = dem.read(1).astype("float64")
			row, col = divmod(i, num_cols)
			ax = axs[row, col]
			dem4slope = rd.rdarray(dem_array, no_data=-9999)
			dem4slope.geotransform = [0, 1, 0, 0, 0, 1,
			                          0]  # defining the geotransfrom, the top left corner is (0,0), width and heigth of a pixel is 1
			# slope = rd.TerrainAttribute(dem4slope, attrib="slope_riserun") #calculate slope
			aspect = rd.TerrainAttribute(dem4slope, attrib="aspect")  # calculate aspect

			handle = rio.plot.show(aspect, transform=dem.transform, ax=ax, title=f"{dem_tif_short_name_list[i]}",
			                       cmap="twilight_shifted", vmin=0, vmax=np.percentile(aspect, 95))  # plot DEM map

			temp_df = group_data
			gdf = gpd.GeoDataFrame(temp_df, geometry=gpd.points_from_xy(temp_df.lon, temp_df.lat), crs=dem.crs)
			gdf.plot(ax=handle, color="red")  # plot location of buildings
			im = handle.get_images()[0]
			cbar = fig.colorbar(im, ax=ax)
			cbar.set_label("Aspect (°)")
			ax.set_xlabel("Longitude(°)")
			ax.set_ylabel("Latitude(°)")

	# Remove any unused subplots
	for i in range(len(axs.flat)):
		if i >= len(dem_tif_path_list):
			fig.delaxes(axs.flat[i])

	# Show the figure
	plt.show()


def main():
	pass


if __name__ == "__main__":
	main()
