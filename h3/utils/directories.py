import os

from h3.utils.file_ops import guarantee_existence
from h3.config import get_h3_dir


def get_data_dir() -> str:
	return guarantee_existence(os.path.join(get_h3_dir(), "data"))
