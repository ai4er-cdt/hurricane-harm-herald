from __future__ import annotations

from h3.utils.downloader import downloader
from h3.utils.directories import get_flood_dir


def _download_flood():
	url = ["https://gis.fema.gov/NFHL/NFHL_Key_Layers.gdb.zip"]  # TODO: put this url somewhere else
	downloader(url, target_dir=get_flood_dir())


def _unpack_flood():
	pass


def main():
	_download_flood()


if __name__ == "__main__":
	main()
