"""Functions to set up h3"""

from h3 import logger
from h3.dataloading.storm_surge import check_storm
from h3.dataloading.terrain_ef import check_dem_files, check_coastlines_file
from h3.dataloading.xbd import get_xbd


def check_all_downloads():
	logger.info("Checking coastline files")
	check_coastlines_file()
	logger.info("Checking DEM files")
	check_dem_files()
	logger.info("Checking xBD files")
	get_xbd()
	logger.info("Checking storm surge files")
	check_storm()
