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


def get_xbd_hlabel_dir(old: bool = False) -> str:
	"""./data/datasets/xBD_data/geotiffs/hold/labels"""
	geotiffs_name = "geotiffs.old" if old else "geotiffs"
	return guarantee_existence(os.path.join(get_xbd_dir(), geotiffs_name, "hold", "labels"))


def get_processed_data_dir() -> str:
	"""./data/datasets/processed_data"""
	return guarantee_existence(os.path.join(get_datasets_dir(), "processed_data"))


def get_metadata_pickle_dir() -> str:
	"""./data/datasets/processed_data/metadata_pickle"""
	return guarantee_existence(os.path.join(get_processed_data_dir(), "metadata_pickle"))
