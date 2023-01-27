import os

from h3.utils.file_ops import guarantee_existence
from h3.config import get_h3_dir


def get_data_dir() -> str:
	return guarantee_existence(os.path.join(get_h3_dir(), "data"))


def get_xbd_dir() -> str:
	return guarantee_existence(os.path.join(get_data_dir(), "xBD_data"))


def get_xbd_disaster_dir(disaster: str) -> str:
	return guarantee_existence(os.path.join(get_xbd_dir(), disaster))


def get_xbd_hurricane_dir() -> str:
	return get_xbd_disaster_dir("hurricane")