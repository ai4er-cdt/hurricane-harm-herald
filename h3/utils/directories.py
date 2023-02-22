import os

from h3.utils.file_ops import guarantee_existence
from h3.config import get_h3_dir


def get_data_dir() -> str:
	"""./data"""
	return guarantee_existence(os.path.join(get_h3_dir(), "data"))


def get_datasets_dir() -> str:
	"""./data/datasets"""
	return guarantee_existence(os.path.join(get_data_dir(), "datasets"))


def get_pickle_dir() -> str:
	"""./data/pickles"""
	return guarantee_existence(os.path.join(get_datasets_dir(), "pickles"))


def get_download_dir() -> str:
	"""./data/downloads"""
	return guarantee_existence(os.path.join(get_data_dir(), "downloads"))


def get_xbd_dir() -> str:
	"""./data/datasets/xBD_data"""
	return guarantee_existence(os.path.join(get_datasets_dir(), "xBD_data"))


def get_xbd_disaster_dir(disaster: str) -> str:
	"""/data/datasets/xBD_data/{disaster}"""
	return guarantee_existence(os.path.join(get_xbd_dir(), disaster))


def get_xbd_hurricane_dir() -> str:
	"""./data/datasets/xBD_data/hurricane"""
	return get_xbd_disaster_dir("hurricane")


def get_weather_data_dir() -> str:
	"""./data/datasets/weather"""
	return guarantee_existence(os.path.join(get_datasets_dir(), "weather"))


def get_cmorph_dir() -> str:
	"""./data/datasets/weather/cmorph"""
	return guarantee_existence(os.path.join(get_weather_data_dir(), "cmorph"))


def get_flood_dir() -> str:
	"""./data/datasets/flood"""
	return guarantee_existence(os.path.join(get_datasets_dir(), "flood"))
