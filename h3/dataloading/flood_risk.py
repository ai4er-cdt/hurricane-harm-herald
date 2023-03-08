from __future__ import annotations

import os

from h3.utils.downloader import downloader
from h3.utils.directories import get_flood_dir
from h3.utils.file_ops import unpack_file


def _download_flood():
	url = ["https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"]  # TODO: put this url somewhere else
	downloader(url, target_dir=get_flood_dir())


def _unpack_flood(filename: str = "NFHL_Key_Layers.gdb.zip"):
	flood_dir = get_flood_dir()
	filepath = os.path.join(flood_dir, filename)
	unpack_file(filepath, "r")


def main():
	_unpack_flood()


if __name__ == "__main__":
	main()
