"""Functions to set up h3"""

from h3 import logger
from h3.dataloading.storm_surge import check_storm
from h3.dataloading.terrain_ef import check_dem_files, check_coastlines_file
from h3.dataloading.xbd import get_xbd
from h3.utils import directories
import os


def check_all_downloads():
	logger.info("Checking coastline files")
	check_coastlines_file()
	logger.info("Checking DEM files")
	check_dem_files()
	logger.info("Checking xBD files")
	get_xbd()
	logger.info("Checking storm surge files")
	check_storm()


def check_if_data_file_exists(
    file_name: str
) -> bool:
    """Fetch data file from location specified in repository README. If it doesn't exist at this location, return an
    error and direct to generate_data_pkls.ipynb to generate necessary file. Files included are:
    - xBD data points. Requires manual download of xBD data and storage in XXX
    - NOAA HURDAT2 Best Track data
    - ECMWF ERA5 data
    - Global hourly ISD data
    - Flood and storm surge risk maps
    - Shortest distance to coast
    - Soil composition (sand, silt, clay)
    """

    # xBD data points
    if file_name == "xbd_points.pkl":
        xbd_points_dir = directories.get_xbd_data_dir()
        file_path = os.path.join(xbd_points_dir, file_name)
    # NOAA HURDAT2 Best Track data (all hurricanes, and xBD hurricanes only)
    elif file_name == "noaa_hurricanes.pkl" or file_name == "noaa_xbd_hurricanes.pkl":
        noaa_dir = directories.get_noaa_data_dir()
        file_path = os.path.join(noaa_dir, file_name)
    # ECMWF ERA5 data
    elif file_name == "era5_xbd_values.pkl":
        ecmwf_dir = directories.get_ecmwf_data_dir()
        file_path = os.path.join(ecmwf_dir, file_name)
    # Global ISD data
    elif file_name == "stations_xbd_values.pkl":
        isd_dir = directories.get_isd_data_dir()
        file_path = os.path.join(isd_dir, file_name)
    else:
        print("todo: non-weather dfs")
        print(f"{file_name} not recognised.")
    # Flood and storm surge risk maps
    # Shortest distance to coast calculation
    # Soil composition data

    if os.path.exists(file_path):
        print(f"{file_name} found at correct location: {file_path}")
        return True
    else:
        print(f"{file_name} not found at correct location: {file_path}")
        return False